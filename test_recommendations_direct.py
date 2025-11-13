"""
Direct test of recommendation function (no API needed)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from app import recommender_service

def test_recommendations():
    """Test recommendations with new fields"""
    print("Testing recommendation function directly...")
    print("="*60)
    
    student_id = "S00001"
    recs = recommender_service.recommend_for_student(student_id, top_k=3)
    
    if not recs:
        print(f"❌ No recommendations found for {student_id}")
        return
    
    print(f"✅ Found {len(recs)} recommendations for {student_id}\n")
    
    print("="*60)
    print("First Recommendation (with all fields):")
    print("="*60)
    
    first_rec = recs[0]
    import json
    print(json.dumps(first_rec, indent=2))
    
    print("\n" + "="*60)
    print("Field Check:")
    print("="*60)
    
    required_fields = ['student_id', 'internship_id', 'title', 'domain', 'score', 'rank']
    new_fields = ['stipend', 'remote', 'state', 'description', 'required_skills', 'min_age', 'max_age', 'capacity']
    
    print("\n✅ Required Fields:")
    for field in required_fields:
        status = "✅" if field in first_rec and first_rec[field] is not None else "❌"
        print(f"   {status} {field}: {first_rec.get(field, 'MISSING')}")
    
    print("\n✅ New Fields (for View Details modal):")
    for field in new_fields:
        status = "✅" if field in first_rec and first_rec[field] is not None else "⚠️"
        value = first_rec.get(field, 'MISSING')
        print(f"   {status} {field}: {value}")
    
    print("\n" + "="*60)
    print("All Recommendations Summary:")
    print("="*60)
    for i, rec in enumerate(recs, 1):
        remote_text = "Remote" if rec.get('remote') == 1 else "On-site"
        stipend = rec.get('stipend', 'N/A')
        state = rec.get('state', 'N/A')
        print(f"{i}. {rec.get('title')}")
        print(f"   Domain: {rec.get('domain')} | Stipend: ₹{stipend} | {remote_text} | State: {state}")
        print(f"   Score: {rec.get('score'):.4f} | Rank: {rec.get('rank')}")
        if rec.get('description'):
            desc = rec.get('description', '')[:60] + "..." if len(rec.get('description', '')) > 60 else rec.get('description', '')
            print(f"   Description: {desc}")
        print()

if __name__ == "__main__":
    test_recommendations()

