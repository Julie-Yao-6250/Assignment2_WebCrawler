#!/usr/bin/env python3
"""
Simple test script to verify sitemap functionality.
"""

from utils.utilities import (
    get_sitemap_urls, parse_sitemap_xml, is_sitemap_url,
    get_robots_checker
)

def test_sitemap_detection():
    """Test sitemap URL detection."""
    print("Testing sitemap URL detection...")
    
    # Test sitemap URL recognition
    test_urls = [
        "http://example.com/sitemap.xml",
        "https://test.uci.edu/sitemap_index.xml", 
        "http://site.com/sitemaps.xml",
        "https://example.com/page.html",
        "http://test.com/sitemap/sitemap.xml"
    ]
    
    for url in test_urls:
        is_sitemap = is_sitemap_url(url)
        print(f"  {url} -> {'✓ sitemap' if is_sitemap else '✗ not sitemap'}")
    
    print()

def test_sitemap_xml_parsing():
    """Test XML sitemap parsing."""
    print("Testing XML sitemap parsing...")
    
    # Test regular sitemap
    sitemap_content = b"""<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
    <url>
        <loc>http://example.com/page1.html</loc>
        <lastmod>2024-01-01</lastmod>
    </url>
    <url>
        <loc>http://example.com/page2.html</loc>
    </url>
</urlset>"""
    
    urls, indexes = parse_sitemap_xml(sitemap_content)
    print(f"  Regular sitemap: {len(urls)} URLs, {len(indexes)} indexes")
    for url in urls:
        print(f"    - {url}")
    
    # Test sitemap index
    index_content = b"""<?xml version="1.0" encoding="UTF-8"?>
<sitemapindex xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
    <sitemap>
        <loc>http://example.com/sitemap1.xml</loc>
    </sitemap>
    <sitemap>
        <loc>http://example.com/sitemap2.xml</loc>
    </sitemap>
</sitemapindex>"""
    
    urls, indexes = parse_sitemap_xml(index_content)
    print(f"  Sitemap index: {len(urls)} URLs, {len(indexes)} indexes")
    for idx in indexes:
        print(f"    - {idx}")
    
    print()

def test_sitemap_discovery():
    """Test sitemap discovery from robots.txt and common locations."""
    print("Testing sitemap discovery...")
    
    test_url = "http://ics.uci.edu"
    sitemap_urls = get_sitemap_urls(test_url)
    
    print(f"  Discovered sitemaps for {test_url}:")
    for sitemap_url in sitemap_urls:
        print(f"    - {sitemap_url}")
    
    if not sitemap_urls:
        print("    (No sitemaps found - this is expected for test)")
    
    print()

def main():
    """Run all tests."""
    print("=== Sitemap Functionality Test ===")
    print()
    
    test_sitemap_detection()
    test_sitemap_xml_parsing()
    test_sitemap_discovery()
    
    print("=== Test Complete ===")

if __name__ == "__main__":
    main()