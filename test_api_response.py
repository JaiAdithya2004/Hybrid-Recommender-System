"""
Quick test to verify API returns internship details
"""

import requests
import json

API_URL = "http://127.0.0.1:8000"

def test_recommendations():
    """Test if recommendations include internship details"""
    print("Testing recommendation endpoint...")
    print("="*60)
    
    try:
        response = requests.get(f"{API_URL}/recommend/S00001?top_k=3", timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            print(f" Success! Got recommendations for: {data['student_id']}")
            print(f"\nNumber of recommendations: {len(data['recommendations'])}")
            
            if data['recommendations']:
                print("\n" + "="*60)
                print("First Recommendation (with all fields):")
                print("="*60)
                first_rec = data['recommendations'][0]
                
                # Pretty print the recommendation
                print(json.dumps(first_rec, indent=2))
                
                print("\n" + "="*60)
                print("Field Check:")
                print("="*60)
                required_fields = ['student_id', 'internship_id', 'title', 'domain', 'score', 'rank']
                new_fields = ['stipend', 'remote', 'state', 'description', 'required_skills']
                
                print("\n✅ Required Fields:")
                for field in required_fields:
                    status = "✅" if field in first_rec and first_rec[field] is not None else "❌"
                    print(f"   {status} {field}: {first_rec.get(field, 'MISSING')}")
                
                print("\n✅ New Fields (for View Details):")
                for field in new_fields:
                    status = "✅" if field in first_rec and first_rec[field] is not None else "⚠️"
                    value = first_rec.get(field, 'MISSING')
                    print(f"   {status} {field}: {value}")
                
                print("\n" + "="*60)
                print("All Recommendations Summary:")
                print("="*60)
                for i, rec in enumerate(data['recommendations'], 1):
                    remote_text = "Remote" if rec.get('remote') == 1 else "On-site"
                    stipend = rec.get('stipend', 'N/A')
                    print(f"{i}. {rec.get('title')} | {rec.get('domain')} | ₹{stipend} | {remote_text} | Score: {rec.get('score'):.4f}")
        else:
            print(f"❌ Error: Status {response.status_code}")
            print(response.text)
            
    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to API. Make sure it's running:")
        print("   uvicorn app.main:app --host 0.0.0.0 --port 8000")
    except Exception as e:
        print(f"❌ Error: {e}")

def test_internship_details():
    """Test the new internship details endpoint"""
    print("\n\n" + "="*60)
    print("Testing internship details endpoint...")
    print("="*60)
    
    try:
        response = requests.get(f"{API_URL}/internship/I00001", timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            print("✅ Success! Got internship details:")
            print(json.dumps(data, indent=2))
        else:
            print(f"❌ Error: Status {response.status_code}")
            print(response.text)
            
    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to API")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    test_recommendations()
    test_internship_details()

