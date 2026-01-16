"""Prompt loader SDK for referencing prompts in agent code."""

from __future__ import annotations

import httpx
from typing import TYPE_CHECKING

from backbone.models.prompt_artifact import PromptVersion

if TYPE_CHECKING:
    from backbone.adapters.event_api import EventEmitter


class PromptLoaderError(Exception):
    """Error loading a prompt from the server."""
    pass


class PromptLoader:
    """Load prompts from the server/storage for use in agent code.
    
    Provides caching to avoid repeated network calls for the same prompt.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        timeout: float = 10.0,
    ) -> None:
        """Initialize the prompt loader.
        
        Args:
            base_url: Base URL of the tracee server
            timeout: HTTP request timeout in seconds
        """
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._cache: dict[tuple[str, str], PromptVersion] = {}

    def _fetch_version(self, prompt_id: str, version_id: str) -> PromptVersion:
        """Fetch a prompt version from the server."""
        cache_key = (prompt_id, version_id)
        
        # Check cache first
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        # Fetch from server
        url = f"{self.base_url}/api/prompts/{prompt_id}/versions/{version_id}"
        
        try:
            with httpx.Client(timeout=self.timeout) as client:
                response = client.get(url)
                
                if response.status_code == 404:
                    raise PromptLoaderError(
                        f"Prompt version not found: {prompt_id}/{version_id}"
                    )
                
                response.raise_for_status()
                data = response.json()
                version = PromptVersion.model_validate(data)
                
                # Cache for future use
                self._cache[cache_key] = version
                return version
                
        except httpx.RequestError as e:
            raise PromptLoaderError(
                f"Failed to connect to server at {self.base_url}: {e}"
            ) from e

    def _fetch_latest(self, prompt_id: str) -> PromptVersion:
        """Fetch the latest version of a prompt from the server."""
        url = f"{self.base_url}/api/prompts/{prompt_id}/latest"
        
        try:
            with httpx.Client(timeout=self.timeout) as client:
                response = client.get(url)
                
                if response.status_code == 404:
                    raise PromptLoaderError(
                        f"Prompt not found or has no versions: {prompt_id}"
                    )
                
                response.raise_for_status()
                data = response.json()
                version = PromptVersion.model_validate(data)
                
                # Cache with actual version ID
                self._cache[(prompt_id, version.version_id)] = version
                return version
                
        except httpx.RequestError as e:
            raise PromptLoaderError(
                f"Failed to connect to server at {self.base_url}: {e}"
            ) from e

    def get_version(
        self,
        prompt_id: str,
        version_id: str = "latest",
    ) -> PromptVersion:
        """Get a full PromptVersion object.
        
        Args:
            prompt_id: The prompt identifier
            version_id: The version to load ("latest" for most recent)
            
        Returns:
            The PromptVersion object with all components
            
        Raises:
            PromptLoaderError: If the prompt/version is not found or server error
        """
        if version_id == "latest":
            return self._fetch_latest(prompt_id)
        return self._fetch_version(prompt_id, version_id)

    def get(
        self,
        prompt_id: str,
        version_id: str = "latest",
    ) -> str:
        """Get resolved prompt text by ID and version.
        
        This is the simplest way to use prompts - just one line of code:
        
            system_prompt = loader.get("planner-prompt", "v2")
        
        Args:
            prompt_id: The prompt identifier
            version_id: The version to load ("latest" for most recent)
            
        Returns:
            The resolved prompt text (all enabled components concatenated)
            
        Raises:
            PromptLoaderError: If the prompt/version is not found or server error
        """
        version = self.get_version(prompt_id, version_id)
        return version.resolve()

    def get_with_trace(
        self,
        prompt_id: str,
        version_id: str,
        agent_id: str,
        emitter: EventEmitter,
    ) -> str:
        """Get resolved prompt text and emit a prompt_resolved trace event.
        
        Use this method when you want the prompt usage to appear in the trace,
        enabling the UI to show which prompt was used by which agent.
        
        Args:
            prompt_id: The prompt identifier
            version_id: The version to load ("latest" for most recent)
            agent_id: The agent loading this prompt
            emitter: The EventEmitter to emit the trace event
            
        Returns:
            The resolved prompt text
            
        Raises:
            PromptLoaderError: If the prompt/version is not found or server error
        """
        version = self.get_version(prompt_id, version_id)
        resolved_text = version.resolve()
        
        # Emit trace event with full prompt snapshot
        emitter.emit_prompt_resolved(
            agent_id=agent_id,
            prompt_id=prompt_id,
            version_id=version.version_id,  # Use actual version ID (in case "latest" was passed)
            resolved_text=resolved_text,
            components=[
                {
                    "type": c.type.value,
                    "content": c.content,
                    "enabled": c.enabled,
                }
                for c in version.components
            ],
            variables_used=version.variables,
        )
        
        return resolved_text

    def clear_cache(self) -> None:
        """Clear the prompt cache.
        
        Useful if prompts have been updated and you need fresh data.
        """
        self._cache.clear()

    def preload(self, prompts: list[tuple[str, str]]) -> None:
        """Preload multiple prompts into the cache.
        
        Args:
            prompts: List of (prompt_id, version_id) tuples to preload
        """
        for prompt_id, version_id in prompts:
            self.get_version(prompt_id, version_id)
