from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class CategoryCompleteness:
    """
    Tracks and verifies completeness metrics for a single category scrape run.
    Ensures that status is never marked 100% complete unless all attempted scope succeeded.
    Note: Completeness is evaluated relative to the requested scope (max_pages, max_listings),
    never claiming whole-website completeness unless exhaustive scraping was conducted.
    """

    category_name: str
    pages_discovered: int = 0
    pages_attempted: int = 0
    pages_scraped: int = 0
    failed_pages: List[Dict[str, Any]] = field(default_factory=list)
    listing_urls_discovered: int = 0
    unique_listing_urls: int = 0
    listings_attempted: int = 0
    listings_scraped: int = 0
    failed_listings: List[Dict[str, Any]] = field(default_factory=list)
    status: str = "COMPLETED"
    max_pages_requested: Optional[int] = None
    max_listings_requested: Optional[int] = None

    @property
    def page_completeness_pct(self) -> float:
        if self.pages_attempted == 0 or self.pages_scraped == 0:
            return 0.0
        return round((self.pages_scraped / self.pages_attempted) * 100.0, 2)

    @property
    def listing_completeness_pct(self) -> float:
        if self.listings_attempted == 0 or self.listings_scraped == 0:
            if self.pages_scraped > 0 and self.unique_listing_urls == 0 and len(self.failed_pages) == 0:
                return 100.0
            return 0.0
        return round((self.listings_scraped / self.listings_attempted) * 100.0, 2)

    @property
    def is_fully_complete(self) -> bool:
        return (
            self.pages_scraped == self.pages_attempted
            and self.pages_scraped > 0
            and len(self.failed_pages) == 0
            and self.listings_scraped == self.listings_attempted
            and len(self.failed_listings) == 0
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "category_name": self.category_name,
            "max_pages_requested": self.max_pages_requested,
            "max_listings_requested": self.max_listings_requested,
            "is_scoped": (
                self.max_pages_requested is not None
                or self.max_listings_requested is not None
            ),
            "pages_discovered": self.pages_discovered,
            "pages_attempted": self.pages_attempted,
            "pages_scraped": self.pages_scraped,
            "failed_pages_count": len(self.failed_pages),
            "page_completeness_pct": self.page_completeness_pct,
            "listing_urls_discovered": self.listing_urls_discovered,
            "unique_listing_urls": self.unique_listing_urls,
            "listings_attempted": self.listings_attempted,
            "listings_scraped": self.listings_scraped,
            "failed_listings_count": len(self.failed_listings),
            "listing_completeness_pct": self.listing_completeness_pct,
            "is_fully_complete": self.is_fully_complete,
            "status": self.status,
        }

    def format_summary(self) -> str:
        lines = [
            f"Category: {self.category_name}",
            f"  Scope:            max_pages={self.max_pages_requested or 'All'}, max_listings={self.max_listings_requested or 'All'}",
            f"  Pages discovered: {self.pages_discovered}",
            f"  Pages attempted:  {self.pages_attempted}",
            f"  Pages scraped:    {self.pages_scraped}",
            f"  Failed pages:     {len(self.failed_pages)}",
            f"  Page Completeness:{self.page_completeness_pct}%",
            f"  Listings found:   {self.listing_urls_discovered} (Unique: {self.unique_listing_urls})",
            f"  Listings scraped: {self.listings_scraped}/{self.listings_attempted}",
            f"  Failed listings:  {len(self.failed_listings)}",
            f"  Listing Comp:     {self.listing_completeness_pct}%",
            f"  Scope Complete:   {'YES' if self.is_fully_complete else 'NO'}",
            f"  Status:           {self.status}",
            "  (Note: Completeness is evaluated within the requested collection scope, not entire website)",
        ]
        return "\n".join(lines)
