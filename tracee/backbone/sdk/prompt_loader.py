"""Prompt loader SDK for referencing prompts in agent code.

so that users can retrieve a prompt they created in the UI with one line of code
system_prompt = loader.get("planner-prompt", "v2")
"""

from __future__ import annotations

import httpx

from backbone.models.prompt_artifact import PromptVersion


class PromptLoaderError(Exception):
    """Exception raised when prompt loading fails."""
    pass


class PromptLoader:
    """Load prompts from the server for use in agent code.
    
    Added caching to avoid repeated network calls for the same prompt.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        timeout: float = 10.0,
    ) -> None:
        """        
        Args:
            base_url: Base URL of the tracee server
            timeout: HTTP request timeout in seconds
        """
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._cache: dict[tuple[str, str], PromptVersion] = {} # key is prompt id and version id

    def _fetch_version(self, prompt_id: str, version_id: str) -> PromptVersion:
        """Fetch a prompt version from the server."""
        cache_key = (prompt_id, version_id)
        
        # check cache first and see if it's a repeated request
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        # fetch from server
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
                
                # cache for future use
                self._cache[cache_key] = version
                return version
                
        except httpx.RequestError as e:
            raise PromptLoaderError(
                f"Failed to connect to server at {self.base_url}: {e}"
            ) from e

    def _fetch_latest(self, prompt_id: str) -> PromptVersion:
        """
        The option to fetch the latest version of a prompt from the server.
        Having this since we store the latest version id with any prompt
        """
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
        """
        if version_id == "latest":
            return self._fetch_latest(prompt_id)
        return self._fetch_version(prompt_id, version_id)

    def get(
        self,
        prompt_id: str,
        version_id: str = "latest",
        agent_id: str | None = None,
    ) -> str:
        """Get resolved prompt text by ID and version.
        Most importantly this works with our TraceEvents 
        so that when a trace is ongoing this will emit an event to the sink so that we know which agent used which prompt version.
        
        Args:
            prompt_id: The prompt identifier
            version_id: The version to load ("latest" for most recent)
            agent_id: Optional agent ID to associate with the trace event
            
        Returns:
            The resolved prompt text (all enabled components concatenated)
        """
        version = self.get_version(prompt_id, version_id)
        resolved_text = version.resolve()
        
        # check the context and emit if tracing is active
        from backbone.sdk.tracing import get_active_context
        ctx = get_active_context()
        if ctx:
            ctx.emit_prompt_resolved(
                prompt_id=prompt_id,
                version_id=version.version_id,
                resolved_text=resolved_text,
                agent_id=agent_id,
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
        """
        self._cache.clear()

    def preload(self, prompts: list[tuple[str, str]]) -> None:
        """Preload multiple prompts into the cache.
        
        Args:
            prompts: List of (prompt_id, version_id) tuples to preload
        """
        for prompt_id, version_id in prompts:
            self.get_version(prompt_id, version_id)
