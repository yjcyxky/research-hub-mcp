"""
GROBID server management using container technology.

This module provides automatic GROBID server management using containers
(Singularity, Podman, or Docker) for reliable PDF text extraction.

Priority order:
1. Singularity (HPC-friendly, creates cached .sif files)
2. Podman (rootless, Docker-compatible)
3. Docker (widely available)
"""

import os
import sys
import time
import shutil
import subprocess
import logging
from pathlib import Path
from typing import Optional, Dict, Any
from enum import Enum

import requests
from tenacity import retry, stop_after_attempt, wait_exponential

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

DEFAULT_IMAGE = "grobid/grobid:0.8.0"

class ContainerRuntime(Enum):
    """Available container runtimes."""
    SINGULARITY = "singularity"
    APPTAINER = "apptainer"  # Singularity successor
    PODMAN = "podman"
    DOCKER = "docker"
    NONE = "none"

    @classmethod
    def detect(cls) -> 'ContainerRuntime':
        """
        Detect available container runtime.

        Checks for container runtimes in priority order:
        1. Singularity/Apptainer
        2. Podman
        3. Docker

        Returns:
            ContainerRuntime enum indicating available runtime
        """
        # Check for Singularity
        if shutil.which("singularity"):
            logger.info("Found Singularity runtime")
            return cls.SINGULARITY

        # Check for Apptainer (Singularity successor)
        if shutil.which("apptainer"):
            logger.info("Found Apptainer runtime")
            return cls.APPTAINER

        # Check for Podman
        if shutil.which("podman"):
            logger.info("Found Podman runtime")
            return cls.PODMAN

        # Check for Docker
        if shutil.which("docker"):
            # Verify Docker daemon is running
            try:
                subprocess.run(["docker", "info"], check=True, capture_output=True, text=True)
                logger.info("Found Docker runtime")
                return cls.DOCKER
            except subprocess.CalledProcessError:
                logger.warning("Docker found but daemon not running")

        return cls.NONE

class GrobidServer:
    """
    GROBID Server Manager using container technology.

    Manages GROBID server lifecycle using containers for reliable
    and reproducible PDF text extraction. Supports Singularity,
    Podman, and Docker runtimes.

    Attributes:
        port: Port number for GROBID server (default: 8070)
        grobid_home: Directory for GROBID cache and data
        image: Container image to use
        container_name: Name for the running container
    """
    DEFAULT_PORT = 8070
    CONTAINER_NAME = "grobid-server"
    SIF_FILENAME = "grobid-0.8.0.sif"

    def __init__(
        self,
        port: int = DEFAULT_PORT,
        host: str = "localhost",
        memory: str = "4g",
        runtime: ContainerRuntime = ContainerRuntime.detect(),
        container_image: str = DEFAULT_IMAGE,
    ):
        """
        Initialize GROBID server manager.

        Args:
            port: Port number for GROBID server
            host: Host address for GROBID server
            memory: Java heap memory allocation
            runtime: Container runtime
            container_image: Container image to use
            image: Container image to use
        """
        self.host = host
        self.memory = memory
        self.runtime = runtime
        self.container_image = container_image

        self.port = port
        self.image = container_image
        self.container_name = f"{self.CONTAINER_NAME}-{port}"

        # Set up GROBID home directory
        self.grobid_home = Path.home() / ".grobid"
        self.grobid_home.mkdir(parents=True, exist_ok=True)

        # Cache directory for Singularity images
        self.cache_dir = self.grobid_home / "cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        if self.runtime == ContainerRuntime.NONE:
            logger.warning(
                "No container runtime found. Please install Singularity, Podman, or Docker.\n"
                "Installation guides:\n"
                "  - Singularity: https://sylabs.io/guides/latest/user-guide/\n"
                "  - Podman: https://podman.io/getting-started/installation\n"
                "  - Docker: https://docs.docker.com/get-docker/"
            )

    def _get_sif_path(self) -> Path:
        """
        Get path to Singularity image file.

        Returns:
            Path to the .sif file
        """
        return self.cache_dir / self.SIF_FILENAME

    def _build_singularity_image(self) -> bool:
        """
        Build Singularity image from Docker image.

        Pulls the Docker image and converts it to Singularity .sif format
        for better performance and caching.

        Returns:
            bool: True if successful, False otherwise
        """
        sif_path = self._get_sif_path()

        # Check if already exists
        if sif_path.exists():
            logger.info(f"Using cached Singularity image: {sif_path}")
            return True

        logger.info(f"Building Singularity image from {self.image}")
        logger.info("This may take a few minutes on first run...")

        cmd = "singularity" if self.runtime == ContainerRuntime.SINGULARITY else "apptainer"

        try:
            # Create a temporary filename for download
            temp_sif = self.cache_dir / "temp_grobid.sif"

            # Pull and convert Docker image to SIF
            result = subprocess.run(
                [cmd, "pull", str(temp_sif), f"docker://{self.image}"],
                check=True,
                capture_output=True,
                text=True
            )

            # Rename to our standard name
            if temp_sif.exists():
                temp_sif.rename(sif_path)
                logger.info(f"Successfully built Singularity image: {sif_path}")
                return True
            else:
                logger.error("Failed to create Singularity image")
                return False

        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to build Singularity image: {e}")
            if e.stderr:
                logger.error(f"Error output: {e.stderr}")
            return False

    def _run_singularity(self) -> bool:
        """
        Run GROBID using Singularity.

        Returns:
            bool: True if successful, False otherwise
        """
        # Build image if needed
        if not self._build_singularity_image():
            return False

        sif_path = self._get_sif_path()
        cmd_name = "singularity" if self.runtime == ContainerRuntime.SINGULARITY else "apptainer"

        # Stop any existing instance
        self.stop_server()

        # Create temp directory for GROBID data
        temp_dir = self.grobid_home / "tmp"
        temp_dir.mkdir(exist_ok=True)

        # Run as an instance for better management
        instance_name = f"grobid-{self.port}"

        cmd = [
            cmd_name, "instance", "start",
            "--bind", f"{temp_dir}:/opt/grobid/grobid-home",
            "--bind", f"{temp_dir}:/tmp",
            "--net", "--network-args", f"portmap={self.port}:8070/tcp",
            str(sif_path),
            instance_name
        ]

        logger.info(f"Starting GROBID with Singularity on port {self.port}")

        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            logger.info(f"Singularity instance started: {instance_name}")

            # Execute GROBID in the instance
            exec_cmd = [
                cmd_name, "exec", "instance://" + instance_name,
                "/opt/grobid/grobid-service/bin/grobid-service"
            ]

            subprocess.Popen(exec_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            return True

        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to start Singularity container: {e}")
            if e.stderr:
                logger.error(f"Error output: {e.stderr}")
            return False

    def _run_podman(self) -> bool:
        """
        Run GROBID using Podman.

        Returns:
            bool: True if successful, False otherwise
        """
        # Stop any existing container
        self.stop_server()

        cmd = [
            "podman", "run",
            "--rm",  # Remove container when stopped
            "-d",    # Detached mode
            "--name", self.container_name,
            "-p", f"{self.port}:8070",
            self.image
        ]

        logger.info(f"Starting GROBID with Podman on port {self.port}")

        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            logger.info(f"GROBID container started: {self.container_name}")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to start Podman container: {e}")
            if e.stderr:
                logger.error(f"Error output: {e.stderr}")
            return False

    def _run_docker(self) -> bool:
        """
        Run GROBID using Docker.

        Returns:
            bool: True if successful, False otherwise
        """
        # Stop any existing container
        self.stop_server()

        cmd = [
            "docker", "run",
            "--rm",  # Remove container when stopped
            "-d",    # Detached mode
            "--name", self.container_name,
            "-p", f"{self.port}:8070",
            self.image
        ]

        logger.info(f"Starting GROBID with Docker on port {self.port}")

        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            logger.info(f"GROBID container started: {self.container_name}")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to start Docker container: {e}")
            if e.stderr:
                logger.error(f"Error output: {e.stderr}")

            # Check if it's a permission issue
            if e.stderr and "permission denied" in e.stderr.lower():
                logger.error(
                    "Permission denied. You may need to:\n"
                    "  1. Run with sudo (not recommended)\n"
                    "  2. Add your user to the docker group: sudo usermod -aG docker $USER\n"
                    "  3. Use Podman instead for rootless containers"
                )
            return False

    def is_running(self) -> bool:
        """
        Check if GROBID server is running.

        Checks both container status and HTTP endpoint availability.

        Returns:
            bool: True if server is running and responsive
        """
        # Check HTTP endpoint first (fastest check)
        try:
            response = requests.get(
                f"http://{self.host}:{self.port}/api/isalive",
                timeout=2
            )
            return response.status_code == 200
        except (requests.RequestException, ConnectionError):
            pass

        # Check container status
        if self.runtime in [ContainerRuntime.PODMAN, ContainerRuntime.DOCKER]:
            cmd_name = self.runtime.value
            try:
                result = subprocess.run(
                    [cmd_name, "ps", "--filter", f"name={self.container_name}"],
                    check=True,
                    capture_output=True,
                    text=True
                )
                return self.container_name in result.stdout
            except subprocess.CalledProcessError:
                pass
        elif self.runtime in [ContainerRuntime.SINGULARITY, ContainerRuntime.APPTAINER]:
            cmd_name = self.runtime.value
            instance_name = f"grobid-{self.port}"
            try:
                result = subprocess.run(
                    [cmd_name, "instance", "list"],
                    check=True,
                    capture_output=True,
                    text=True
                )
                return instance_name in result.stdout
            except subprocess.CalledProcessError:
                pass

        return False

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    def _wait_for_server(self, timeout: int = 60) -> bool:
        """
        Wait for GROBID server to become responsive.

        Args:
            timeout: Maximum seconds to wait

        Returns:
            bool: True if server is responsive, False if timeout
        """
        logger.info(f"Waiting for GROBID server to start on port {self.port}...")

        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                response = requests.get(
                    f"http://{self.host}:{self.port}/api/isalive",
                    timeout=2
                )
                if response.status_code == 200:
                    logger.info("GROBID server is ready!")
                    return True
            except (requests.RequestException, ConnectionError):
                pass

            time.sleep(2)

        logger.error(f"GROBID server failed to start within {timeout} seconds")
        return False

    def start_server(self, force: bool = False, timeout: int = 60) -> bool:
        """
        Start GROBID server using appropriate container runtime.

        Args:
            force: Force restart even if already running
            timeout: Maximum seconds to wait for server startup

        Returns:
            bool: True if server started successfully
        """
        # Check if already running
        if not force and self.is_running():
            logger.info(f"GROBID server already running on port {self.port}")
            return True

        # Check runtime availability
        if self.runtime == ContainerRuntime.NONE:
            logger.error("No container runtime available")
            return False

        # Stop existing server if force restart
        if force:
            self.stop_server()

        # Start with appropriate runtime
        success = False

        if self.runtime in [ContainerRuntime.SINGULARITY, ContainerRuntime.APPTAINER]:
            success = self._run_singularity()
        elif self.runtime == ContainerRuntime.PODMAN:
            success = self._run_podman()
        elif self.runtime == ContainerRuntime.DOCKER:
            success = self._run_docker()

        if not success:
            return False

        # Wait for server to be responsive
        return self._wait_for_server(timeout)

    def stop_server(self) -> bool:
        """
        Stop GROBID server.

        Returns:
            bool: True if stopped successfully or not running
        """
        if self.runtime == ContainerRuntime.NONE:
            return True

        if self.runtime in [ContainerRuntime.SINGULARITY, ContainerRuntime.APPTAINER]:
            cmd_name = self.runtime.value
            instance_name = f"grobid-{self.port}"

            try:
                # Stop the Singularity instance
                subprocess.run(
                    [cmd_name, "instance", "stop", instance_name],
                    capture_output=True,
                    text=True
                )
                logger.info(f"Stopped Singularity instance: {instance_name}")
                return True
            except Exception as e:
                logger.debug(f"Failed to stop Singularity instance: {e}")
                return True  # Probably wasn't running

        # For Podman/Docker
        if self.runtime in [ContainerRuntime.PODMAN, ContainerRuntime.DOCKER]:
            cmd_name = self.runtime.value

            try:
                # Check if container exists
                result = subprocess.run(
                    [cmd_name, "ps", "-a", "--filter", f"name={self.container_name}"],
                    check=True,
                    capture_output=True,
                    text=True
                )

                if self.container_name in result.stdout:
                    # Stop container
                    subprocess.run(
                        [cmd_name, "stop", self.container_name],
                        check=True,
                        capture_output=True,
                        text=True
                    )

                    # Remove container
                    subprocess.run(
                        [cmd_name, "rm", self.container_name],
                        capture_output=True,
                        text=True
                    )

                    logger.info(f"Stopped GROBID container: {self.container_name}")

                return True

            except subprocess.CalledProcessError as e:
                logger.debug(f"Failed to stop container: {e}")
                return True  # Probably wasn't running

        return True

    def restart_server(self, timeout: int = 60) -> bool:
        """
        Restart GROBID server.

        Args:
            timeout: Maximum seconds to wait for server startup

        Returns:
            bool: True if restarted successfully
        """
        logger.info("Restarting GROBID server...")
        self.stop_server()
        time.sleep(2)
        return self.start_server(timeout=timeout)

    def get_server_info(self) -> Dict[str, Any]:
        """
        Get GROBID server information.

        Returns:
            Dictionary containing server status and configuration
        """
        info = {
            "runtime": self.runtime.value,
            "port": self.port,
            "url": f"http://{self.host}:{self.port}",
            "image": self.image,
            "container_name": self.container_name,
            "grobid_home": str(self.grobid_home),
            "is_running": self.is_running()
        }

        if self.runtime in [ContainerRuntime.SINGULARITY, ContainerRuntime.APPTAINER]:
            info["sif_path"] = str(self._get_sif_path())
            info["sif_exists"] = self._get_sif_path().exists()

        return info

    def cleanup(self) -> None:
        """
        Clean up resources and stop server.
        """
        self.stop_server()
        logger.info("GROBID server cleanup completed")

    def __enter__(self):
        """Context manager entry."""
        self.start_server()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.cleanup()


def ensure_grobid_server(
    port: int = 8070,
    host: str = "localhost",
    memory: str = "4g",
    runtime: ContainerRuntime = ContainerRuntime.detect(),
    container_image: str = DEFAULT_IMAGE,
    timeout: int = 60
) -> Optional[str]:
    """
    Ensure GROBID server is running and return its URL.

    This is a convenience function that creates a GrobidServer instance,
    starts it if needed, and returns the URL.

    Args:
        port: Port number for GROBID server
        grobid_home: Directory for GROBID cache and data
        timeout: Maximum seconds to wait for server startup

    Returns:
        GROBID server URL if successful, None otherwise

    Examples:
        >>> url = ensure_grobid_server()
        >>> if url:
        ...     print(f"GROBID running at {url}")
    """
    try:
        server = GrobidServer(
            port=port,
            host=host,
            memory=memory,
            runtime=runtime,
            container_image=container_image
        )

        if server.runtime == ContainerRuntime.NONE:
            logger.error("No container runtime found")
            return None

        if server.start_server(timeout=timeout):
            return f"http://{server.host}:{port}"

        return None

    except Exception as e:
        logger.error(f"Failed to ensure GROBID server: {e}")
        return None
