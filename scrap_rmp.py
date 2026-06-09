"""
scrape_rmp.py
-------------
Scrapes UCLA professor data + individual review text from Rate My Professor.

Strategy:
  - RMP GraphQL API -> school lookup, professor list with aggregate stats,
    and individual review comments per professor.

Note: the RateMyProfessorAPI PyPI package no longer works — RMP blocks its
header-less HTML requests with HTTP 403. The GraphQL endpoint, however, still
responds when given a browser User-Agent and RMP's public "test:test" token,
so we use it for everything here.

Output: documents/rmp_raw.json  (one record per review)
"""

import json
import time
import base64
import requests
from pathlib import Path

#Config
UCLA_SCHOOL_ID = 1075
MAX_PROFESSORS = 60
REVIEWS_PER_PROF = 10
DELAY_SECONDS = 1.0
MIN_RATINGS = 5
PROFESSOR_FETCH_LIMIT = 1000
OUTPUT_PATH = Path("documents/rmp_raw.json")

#RMP GraphQL endpoint + headers
GRAPHQL_URL = "https://www.ratemyprofessors.com/graphql"
GRAPHQL_HEADERS = {
    "Content-Type": "application/json",
    "User-Agent": "Mozilla/5.0",
    "Authorization": "Basic dGVzdDp0ZXN0", 
}

#GraphQL query: search professors within a school (paginated)
TEACHERS_QUERY = """
query TeacherSearchQuery($query: TeacherSearchQuery!, $count: Int!, $cursor: String) {
  newSearch {
    teachers(query: $query, first: $count, after: $cursor) {
      pageInfo { hasNextPage endCursor }
      edges {
        node {
          id
          legacyId
          firstName
          lastName
          department
          avgRating
          avgDifficulty
          numRatings
          wouldTakeAgainPercent
        }
      }
    }
  }
}
"""

#GraphQL query: individual reviews for one professor
REVIEWS_QUERY = """
query TeacherRatingsPageQuery($id: ID!, $count: Int!) {
  node(id: $id) {
    ... on Teacher {
      ratings(first: $count) {
        edges {
          node {
            comment
            class
            date
            helpfulRating
            clarityRating
            difficultyRating
            wouldTakeAgain
            grade
            attendanceMandatory
            ratingTags
          }
        }
      }
    }
  }
}
"""


#Helpers
def graphql(query: str, variables: dict) -> dict:
    """POST a GraphQL query and return the parsed `data` object (or {})."""
    resp = requests.post(
        GRAPHQL_URL,
        json={"query": query, "variables": variables},
        headers=GRAPHQL_HEADERS,
        timeout=15,
    )
    resp.raise_for_status()
    body = resp.json()
    if body.get("errors"):
        raise RuntimeError(f"GraphQL errors: {body['errors']}")
    return body.get("data", {}) or {}


def school_node_id(legacy_id: int) -> str:
    """RMP node IDs are base64 of "School-{id}", e.g. School-1075 -> U2Nob29sLTEwNzU=."""
    return base64.b64encode(f"School-{legacy_id}".encode()).decode()


def fetch_professors(school_legacy_id: int, limit: int) -> list[dict]:
    """Fetch up to `limit` professors for a school, following pagination."""
    school_id = school_node_id(school_legacy_id)
    profs: list[dict] = []
    cursor = None

    while len(profs) < limit:
        data = graphql(
            TEACHERS_QUERY,
            {
                "query": {"text": "", "schoolID": school_id},
                "count": min(100, limit - len(profs)),
                "cursor": cursor,
            },
        )
        teachers = data.get("newSearch", {}).get("teachers", {})
        edges = teachers.get("edges", [])
        if not edges:
            break

        for e in edges:
            node = e.get("node")
            if not node:
                continue
            first = (node.get("firstName") or "").strip()
            last = (node.get("lastName") or "").strip()
            profs.append(
                {
                    "node_id": node["id"],
                    "legacy_id": node.get("legacyId"),
                    "name": f"{first} {last}".strip(),
                    "department": node.get("department") or "",
                    "rating": node.get("avgRating"),
                    "difficulty": node.get("avgDifficulty"),
                    "num_ratings": node.get("numRatings") or 0,
                    "would_take_again": node.get("wouldTakeAgainPercent"),
                }
            )

        page = teachers.get("pageInfo", {})
        if not page.get("hasNextPage"):
            break
        cursor = page.get("endCursor")
        time.sleep(DELAY_SECONDS)

    return profs


def fetch_reviews(node_id: str, count: int = REVIEWS_PER_PROF) -> list[dict]:
    """Fetch individual review records for a professor via GraphQL."""
    try:
        data = graphql(REVIEWS_QUERY, {"id": node_id, "count": count})
        edges = data.get("node", {}).get("ratings", {}).get("edges", [])
        return [e["node"] for e in edges if e.get("node")]
    except Exception as e:
        print(f"    [!] GraphQL error for {node_id}: {e}")
        return []


def build_document(prof: dict, review: dict) -> dict | None:
    """
    Combine professor metadata + one review into our standard document schema.
    The 'text' field is what gets chunked and embedded downstream.
    """
    comment = (review.get("comment") or "").strip()
    if not comment or comment.lower() in ("no comments", "n/a"):
        return None

    would_take_again = prof.get("would_take_again")

    # Human-readable text blob
    text_parts = [
        f"Professor: {prof['name']}",
        f"Department: {prof['department']}",
        f"School: UCLA",
        f"Overall rating: {prof['rating']} / 5.0",
        f"Difficulty: {prof['difficulty']} / 5.0",
    ]
    if would_take_again is not None and would_take_again >= 0:
        text_parts.append(f"Would take again: {round(would_take_again, 1)}%")
    if review.get("class"):
        text_parts.append(f"Course: {review['class']}")
    if review.get("grade"):
        text_parts.append(f"Grade received: {review['grade']}")
    text_parts.append(f"Student review: {comment}")

    return {
        "text": "\n".join(text_parts),
        "source": "ratemyprofessor",
        "url": f"https://www.ratemyprofessors.com/professor/{prof['legacy_id']}",
        "date": review.get("date", "")[:10] if review.get("date") else "",
        "topic": "professors",
        "metadata": {
            "professor": prof["name"],
            "department": prof["department"],
            "overall_rating": prof["rating"],
            "difficulty": prof["difficulty"],
            "would_take_again": would_take_again,
            "course": review.get("class", ""),
            "grade": review.get("grade", ""),
            "tags": review.get("ratingTags", ""),
            "helpful_rating": review.get("helpfulRating"),
            "clarity_rating": review.get("clarityRating"),
            "would_take_again_review": review.get("wouldTakeAgain"),
        },
    }


def main():
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    print("Fetching UCLA professors from RMP GraphQL...")
    professors = fetch_professors(UCLA_SCHOOL_ID, PROFESSOR_FETCH_LIMIT)
    print(f"Found {len(professors)} professors total.")

    #Filter to those with enough ratings to be useful
    professors = [p for p in professors if p["num_ratings"] >= MIN_RATINGS]
    print(f"{len(professors)} professors with {MIN_RATINGS}+ ratings.")

    #Sort by rating count descending — most-reviewed profs first
    professors.sort(key=lambda p: p["num_ratings"], reverse=True)
    professors = professors[:MAX_PROFESSORS]
    print(f"Processing top {len(professors)} professors...\n")

    documents = []
    skipped_reviews = 0

    for i, prof in enumerate(professors, 1):
        print(f"[{i}/{len(professors)}] {prof['name']} ({prof['department']}) — {prof['num_ratings']} ratings")

        reviews = fetch_reviews(prof["node_id"], count=REVIEWS_PER_PROF)
        print(f"    Retrieved {len(reviews)} reviews via GraphQL")

        for review in reviews:
            doc = build_document(prof, review)
            if doc:
                documents.append(doc)
            else:
                skipped_reviews += 1

        time.sleep(DELAY_SECONDS)

    # Save all documents to JSON file
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(documents, f, indent=2, ensure_ascii=False)

    print(f"\nDone.")
    print(f"  Documents saved : {len(documents)}")
    print(f"  Reviews skipped : {skipped_reviews} (empty/no comment)")
    print(f"  Output          : {OUTPUT_PATH}")

    if documents:
        print("\nExample document:")
        print(json.dumps(documents[0], indent=2))


if __name__ == "__main__":
    main()
