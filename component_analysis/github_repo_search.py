#!/usr/bin/env python3

import json
import os
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import requests


SEARCH_QUERIES = [
    "multi-agent LLM",
    "multi agent LLM",
    "multi-agent system LLM",
    "LLM multi agent",
    "autonomous agents application",
    "collaborative agents",

]

# GitHub API settings
GITHUB_API_BASE = "https://api.github.com"
RESULTS_PER_PAGE = 100
MAX_PAGES_PER_QUERY = 10
MIN_STARS = 100

REQUESTS_PER_MINUTE = 30
REQUEST_DELAY = 60 / REQUESTS_PER_MINUTE



@dataclass
class RepositoryInfo:
    """Repository metadata container."""
    full_name: str  # owner/repo format
    name: str
    owner: str
    description: Optional[str]
    url: str
    html_url: str
    stars: int
    forks: int
    watchers: int
    open_issues: int
    topics: list[str]
    language: Optional[str]
    created_at: str
    updated_at: str
    pushed_at: str
    license: Optional[str]
    default_branch: str
    size_kb: int
    is_fork: bool
    is_archived: bool
    matched_queries: list[str]  # Which search queries found this repo


@dataclass
class SearchResult:
    """Container for search results and metadata."""
    repositories: list[RepositoryInfo]
    total_found: int
    queries_executed: list[str]
    search_timestamp: str
    rate_limit_remaining: Optional[int]
    errors: list[str]



class GitHubSearcher:
    
    def __init__(self, token: Optional[str] = None):
        self.token = token or os.environ.get("GITHUB_TOKEN")
        self.session = requests.Session()
        
        if self.token:
            self.session.headers["Authorization"] = f"token {self.token}"
            print("✓ Using authenticated requests (higher rate limits)")
        else:
            print("⚠ No GitHub token found. Using unauthenticated requests (lower rate limits)")
            print("  Set GITHUB_TOKEN environment variable for better performance")
        
        self.session.headers["Accept"] = "application/vnd.github.v3+json"
        self.session.headers["User-Agent"] = "MAS-Component-Analysis"
        
        self._last_request_time = 0
        self._rate_limit_remaining = None
        self._rate_limit_reset = None
        
        # Track seen repositories for deduplication
        self._seen_repos: set[str] = set()
        self._errors: list[str] = []
    
    def _respect_rate_limit(self):
        """Ensure we don't exceed rate limits."""
        # Check if we need to wait for rate limit reset
        if self._rate_limit_remaining is not None and self._rate_limit_remaining < 5:
            if self._rate_limit_reset:
                wait_time = max(0, self._rate_limit_reset - time.time()) + 1
                if wait_time > 0:
                    print(f"  ⏳ Rate limit low, waiting {wait_time:.0f}s for reset...")
                    time.sleep(wait_time)
        
        # Enforce minimum delay between requests
        elapsed = time.time() - self._last_request_time
        if elapsed < REQUEST_DELAY:
            time.sleep(REQUEST_DELAY - elapsed)
        
        self._last_request_time = time.time()
    
    def _update_rate_limit_info(self, response: requests.Response):
        """Update rate limit tracking from response headers."""
        if "X-RateLimit-Remaining" in response.headers:
            self._rate_limit_remaining = int(response.headers["X-RateLimit-Remaining"])
        if "X-RateLimit-Reset" in response.headers:
            self._rate_limit_reset = int(response.headers["X-RateLimit-Reset"])
    
    def _parse_repository(self, repo_data: dict, query: str) -> Optional[RepositoryInfo]:
        """Parse repository data from API response."""
        try:
            return RepositoryInfo(
                full_name=repo_data["full_name"],
                name=repo_data["name"],
                owner=repo_data["owner"]["login"],
                description=repo_data.get("description"),
                url=repo_data["url"],
                html_url=repo_data["html_url"],
                stars=repo_data["stargazers_count"],
                forks=repo_data["forks_count"],
                watchers=repo_data["watchers_count"],
                open_issues=repo_data["open_issues_count"],
                topics=repo_data.get("topics", []),
                language=repo_data.get("language"),
                created_at=repo_data["created_at"],
                updated_at=repo_data["updated_at"],
                pushed_at=repo_data["pushed_at"],
                license=repo_data["license"]["spdx_id"] if repo_data.get("license") else None,
                default_branch=repo_data["default_branch"],
                size_kb=repo_data["size"],
                is_fork=repo_data["fork"],
                is_archived=repo_data["archived"],
                matched_queries=[query],
            )
        except KeyError as e:
            self._errors.append(f"Failed to parse repo data: missing key {e}")
            return None
    
    def search_repositories(
        self,
        query: str,
        min_stars: int = MIN_STARS,
        max_pages: int = MAX_PAGES_PER_QUERY,
    ) -> list[RepositoryInfo]:
        """
        Args:
            query: Search query string
            min_stars: Minimum star count filter
            max_pages: Maximum pages to fetch
            
        Returns:
            List of repository info objects
        """
        repositories = []
        
        # Calculate date 24 months ago (approx 730 days)
        cutoff_date = (datetime.now() - timedelta(days=730)).strftime("%Y-%m-%d")

        # exclusions = (
        #     "-tutorial -course -education -learn -lecture "
        #     "-starter -boilerplate -awesome-list -collection"
        # )
        
        # Build search query with filters
        # search_query = f"{query} stars:>={min_stars}"
        # search_query = f"{query} {exclusions} in:readme stars:>={min_stars} pushed:>={cutoff_date}"
        search_query = f"{query} in:readme stars:>={min_stars} pushed:>={cutoff_date}"
        
        for page in range(1, max_pages + 1):
            self._respect_rate_limit()
            
            params = {
                "q": search_query,
                "sort": "stars",
                "order": "desc",
                "per_page": RESULTS_PER_PAGE,
                "page": page,
            }
            
            try:
                response = self.session.get(
                    f"{GITHUB_API_BASE}/search/repositories",
                    params=params,
                    timeout=30,
                )
                self._update_rate_limit_info(response)
                
                if response.status_code == 403:
                    # Rate limited
                    self._errors.append(f"Rate limited on query '{query}' page {page}")
                    print(f"  ⚠ Rate limited, stopping query")
                    break
                
                if response.status_code == 422:
                    # Validation error (e.g., query too complex)
                    self._errors.append(f"Query validation error: {query}")
                    break
                
                response.raise_for_status()
                data = response.json()
                
                items = data.get("items", [])
                if not items:
                    break  # No more results
                
                for item in items:
                    repo_info = self._parse_repository(item, query)
                    if repo_info:
                        repositories.append(repo_info)
                
                # Check if we've fetched all available results
                total_count = data.get("total_count", 0)
                if page * RESULTS_PER_PAGE >= total_count:
                    break
                    
            except requests.RequestException as e:
                self._errors.append(f"Request error for query '{query}': {str(e)}")
                print(f"  ⚠ Request error: {e}")
                break
        
        return repositories
    
    def search_all_queries(
        self,
        queries: list[str] = SEARCH_QUERIES,
        min_stars: int = MIN_STARS,
    ) -> SearchResult:
        """
        Execute all search queries with deduplication.
        
        Args:
            queries: List of search query strings
            min_stars: Minimum star count filter
            
        Returns:
            SearchResult containing all unique repositories found
        """
        all_repos: dict[str, RepositoryInfo] = {}  # full_name -> repo info
        
        print(f"\n{'='*60}")
        print(f"Starting GitHub search with {len(queries)} queries")
        print(f"Minimum stars filter: {min_stars}")
        print(f"{'='*60}\n")
        
        for i, query in enumerate(queries, 1):
            print(f"[{i}/{len(queries)}] Searching: '{query}'")
            
            repos = self.search_repositories(query, min_stars=min_stars)
            
            new_count = 0
            for repo in repos:
                if repo.full_name not in all_repos:
                    all_repos[repo.full_name] = repo
                    self._seen_repos.add(repo.full_name)
                    new_count += 1
                else:
                    # Add this query to matched_queries for deduplication tracking
                    all_repos[repo.full_name].matched_queries.append(query)
            
            print(f"  Found {len(repos)} repos, {new_count} new unique")
            
            if self._rate_limit_remaining:
                print(f"  Rate limit remaining: {self._rate_limit_remaining}")
        
        # Sort by stars descending
        sorted_repos = sorted(all_repos.values(), key=lambda r: r.stars, reverse=True)
        
        result = SearchResult(
            repositories=sorted_repos,
            total_found=len(sorted_repos),
            queries_executed=queries,
            search_timestamp=datetime.utcnow().isoformat() + "Z",
            rate_limit_remaining=self._rate_limit_remaining,
            errors=self._errors,
        )
        
        print(f"\n{'='*60}")
        print(f"Search complete!")
        print(f"Total unique repositories found: {len(sorted_repos)}")
        print(f"Errors encountered: {len(self._errors)}")
        print(f"{'='*60}\n")
        
        return result


# def save_results_json(result: SearchResult, output_path: Path):
#     """Save search results to JSON file."""
#     output_data = {
#         "metadata": {
#             "search_timestamp": result.search_timestamp,
#             "total_repositories": result.total_found,
#             "queries_executed": result.queries_executed,
#             "rate_limit_remaining": result.rate_limit_remaining,
#             "errors": result.errors,
#         },
#         "repositories": [asdict(repo) for repo in result.repositories],
#     }
    
#     with open(output_path, "w", encoding="utf-8") as f:
#         json.dump(output_data, f, indent=2, ensure_ascii=False)
    
#     print(f"✓ Results saved to: {output_path}")


def save_results_csv(result: SearchResult, output_path: Path):
    """Save search results to CSV file for easy viewing."""
    import csv
    
    fieldnames = [
        "full_name", "stars", "forks", "language", "description",
        "topics", "updated_at", "license", "html_url", "matched_queries_count"
    ]
    
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        for repo in result.repositories:
            writer.writerow({
                "full_name": repo.full_name,
                "stars": repo.stars,
                "forks": repo.forks,
                "language": repo.language or "",
                "description": (repo.description or "")[:200],  # Truncate long descriptions
                "topics": ", ".join(repo.topics),
                "updated_at": repo.updated_at,
                "license": repo.license or "",
                "html_url": repo.html_url,
                "matched_queries_count": len(repo.matched_queries),
            })
    
    print(f"✓ CSV export saved to: {output_path}")

def main():
    # Initialize searcher (will use GITHUB_TOKEN env var if available)
    searcher = GitHubSearcher()
    
    # Execute search
    result = searcher.search_all_queries(
        queries=SEARCH_QUERIES,
        min_stars=MIN_STARS,
    )
    
    # Define output paths
    output_dir = Path(__file__).parent / "data"
    output_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = output_dir / f"github_repos_{timestamp}.json"
    csv_path = output_dir / f"github_repos_{timestamp}.csv"
    
    # save_results_json(result, json_path)
    save_results_csv(result, csv_path)
    
    # print_summary(result)
    
    # Also save a "latest" copy for easy access
    latest_json = output_dir / "github_repos_latest.json"
    latest_csv = output_dir / "github_repos_latest.csv"
    # save_results_json(result, latest_json)
    save_results_csv(result, latest_csv)
    
    return result


if __name__ == "__main__":
    main()

