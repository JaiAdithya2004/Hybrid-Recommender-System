"""
Performance Testing Script for Hybrid Recommender System
Tests recommendation quality, allocation efficiency, and API performance
"""

import time
import requests
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict
import json
from datetime import datetime

# Configuration
API_BASE_URL = "http://127.0.0.1:8000"  # Change if deployed
OUT_DIR = Path("notebook/outputs_recommender_v2")
RESULTS_DIR = Path("performance_results")
RESULTS_DIR.mkdir(exist_ok=True)

class PerformanceTester:
    def __init__(self, api_url: str = API_BASE_URL, out_dir: Path = OUT_DIR):
        self.api_url = api_url
        self.out_dir = out_dir
        self.results = {}
        
    def load_data(self):
        """Load recommendations, allocations, and metadata"""
        print("Loading data files...")
        self.recommendations_df = pd.read_csv(self.out_dir / "recommendations.csv")
        self.allocations_df = pd.read_csv(self.out_dir / "allocations.csv")
        self.students_df = pd.read_csv(self.out_dir / "students_synthetic.csv")
        self.internships_df = pd.read_csv(self.out_dir / "internships_synthetic.csv")
        self.metrics_df = pd.read_csv(self.out_dir / "evaluation_metrics.csv")
        print(f"✓ Loaded {len(self.recommendations_df)} recommendations")
        print(f"✓ Loaded {len(self.allocations_df)} allocations")
        print(f"✓ Loaded {len(self.students_df)} students")
        print(f"✓ Loaded {len(self.internships_df)} internships")
        
    def test_recommendation_quality(self) -> Dict:
        """Test recommendation quality metrics"""
        print("\n" + "="*60)
        print("TESTING RECOMMENDATION QUALITY")
        print("="*60)
        
        metrics = {}
        
        # 1. Score Distribution
        scores = self.recommendations_df['score'].values
        metrics['score_mean'] = float(np.mean(scores))
        metrics['score_std'] = float(np.std(scores))
        metrics['score_min'] = float(np.min(scores))
        metrics['score_max'] = float(np.max(scores))
        metrics['score_median'] = float(np.median(scores))
        print(f"\n📊 Score Statistics:")
        print(f"   Mean: {metrics['score_mean']:.4f}")
        print(f"   Std:  {metrics['score_std']:.4f}")
        print(f"   Min:  {metrics['score_min']:.4f}")
        print(f"   Max:  {metrics['score_max']:.4f}")
        print(f"   Median: {metrics['score_median']:.4f}")
        
        # 2. Coverage Metrics
        unique_students = self.recommendations_df['student_id'].nunique()
        unique_internships = self.recommendations_df['internship_id'].nunique()
        total_students = len(self.students_df)
        total_internships = len(self.internships_df)
        
        metrics['student_coverage'] = unique_students / total_students
        metrics['internship_coverage'] = unique_internships / total_internships
        metrics['catalog_coverage'] = unique_internships / total_internships
        
        print(f"\n📈 Coverage Metrics:")
        print(f"   Student Coverage: {metrics['student_coverage']*100:.2f}% ({unique_students}/{total_students})")
        print(f"   Internship Coverage: {metrics['internship_coverage']*100:.2f}% ({unique_internships}/{total_internships})")
        
        # 3. Diversity Metrics
        # Intra-list diversity (average pairwise distance between recommended items per student)
        diversity_scores = []
        for student_id in self.recommendations_df['student_id'].unique()[:100]:  # Sample for speed
            student_recs = self.recommendations_df[
                self.recommendations_df['student_id'] == student_id
            ].head(10)
            if len(student_recs) > 1:
                domains = student_recs['domain'].values
                unique_domains = len(set(domains))
                diversity_scores.append(unique_domains / len(domains))
        
        metrics['diversity_mean'] = float(np.mean(diversity_scores)) if diversity_scores else 0.0
        print(f"\n🎯 Diversity Metrics:")
        print(f"   Average Domain Diversity: {metrics['diversity_mean']:.4f}")
        
        # 4. Popularity Bias
        internship_counts = self.recommendations_df['internship_id'].value_counts()
        metrics['popularity_gini'] = self._calculate_gini(internship_counts.values)
        print(f"   Popularity Gini Coefficient: {metrics['popularity_gini']:.4f} (lower = less bias)")
        
        # 5. Precision@K and Recall@K (if we had ground truth)
        # For now, we'll use score thresholds
        high_score_threshold = np.percentile(scores, 75)
        high_score_count = (scores >= high_score_threshold).sum()
        metrics['high_score_ratio'] = high_score_count / len(scores)
        print(f"\n⭐ Quality Metrics:")
        print(f"   High Score Ratio (≥75th percentile): {metrics['high_score_ratio']*100:.2f}%")
        
        # 6. Rank Distribution
        rank_counts = self.recommendations_df['rank'].value_counts().sort_index()
        metrics['rank_distribution'] = rank_counts.to_dict()
        print(f"\n📋 Rank Distribution:")
        for rank, count in rank_counts.head(10).items():
            print(f"   Rank {rank}: {count} recommendations")
        
        self.results['recommendation_quality'] = metrics
        return metrics
    
    def test_allocation_performance(self) -> Dict:
        """Test allocation efficiency and fairness"""
        print("\n" + "="*60)
        print("TESTING ALLOCATION PERFORMANCE")
        print("="*60)
        
        metrics = {}
        
        # 1. Allocation Statistics
        total_allocations = len(self.allocations_df)
        total_capacity = self.internships_df['capacity'].sum()
        metrics['allocation_count'] = total_allocations
        metrics['total_capacity'] = int(total_capacity)
        metrics['utilization_rate'] = total_allocations / total_capacity if total_capacity > 0 else 0.0
        
        print(f"\n📊 Allocation Statistics:")
        print(f"   Total Allocations: {total_allocations}")
        print(f"   Total Capacity: {int(total_capacity)}")
        print(f"   Utilization Rate: {metrics['utilization_rate']*100:.2f}%")
        
        # 2. Capacity Utilization per Internship
        allocation_counts = self.allocations_df['internship_id'].value_counts()
        utilization_by_internship = []
        for internship_id in self.internships_df['internship_id']:
            allocated = allocation_counts.get(internship_id, 0)
            capacity = self.internships_df[
                self.internships_df['internship_id'] == internship_id
            ]['capacity'].values[0]
            if capacity > 0:
                utilization_by_internship.append(allocated / capacity)
        
        metrics['avg_internship_utilization'] = float(np.mean(utilization_by_internship))
        metrics['utilization_std'] = float(np.std(utilization_by_internship))
        print(f"\n📈 Capacity Utilization:")
        print(f"   Average: {metrics['avg_internship_utilization']*100:.2f}%")
        print(f"   Std Dev: {metrics['utilization_std']*100:.2f}%")
        
        # 3. Allocation Score Statistics
        alloc_scores = self.allocations_df['score'].values
        metrics['alloc_score_mean'] = float(np.mean(alloc_scores))
        metrics['alloc_score_std'] = float(np.std(alloc_scores))
        metrics['alloc_score_min'] = float(np.min(alloc_scores))
        metrics['alloc_score_max'] = float(np.max(alloc_scores))
        print(f"\n⭐ Allocation Score Statistics:")
        print(f"   Mean: {metrics['alloc_score_mean']:.4f}")
        print(f"   Std:  {metrics['alloc_score_std']:.4f}")
        print(f"   Range: [{metrics['alloc_score_min']:.4f}, {metrics['alloc_score_max']:.4f}]")
        
        # 4. Fairness Metrics
        # Check distribution across student demographics
        merged = self.allocations_df.merge(
            self.students_df[['student_id', 'rural', 'female']],
            on='student_id',
            how='left'
        )
        
        if 'rural' in merged.columns:
            rural_allocations = merged['rural'].sum()
            total_rural = self.students_df['rural'].sum()
            metrics['rural_allocation_rate'] = rural_allocations / total_rural if total_rural > 0 else 0.0
            print(f"\n⚖️  Fairness Metrics:")
            print(f"   Rural Student Allocation Rate: {metrics['rural_allocation_rate']*100:.2f}%")
        
        if 'female' in merged.columns:
            female_allocations = merged['female'].sum()
            total_female = self.students_df['female'].sum()
            metrics['female_allocation_rate'] = female_allocations / total_female if total_female > 0 else 0.0
            print(f"   Female Student Allocation Rate: {metrics['female_allocation_rate']*100:.2f}%")
        
        # 5. Domain Distribution
        alloc_domains = self.allocations_df.merge(
            self.internships_df[['internship_id', 'domain']],
            on='internship_id',
            how='left'
        )['domain'].value_counts()
        metrics['domain_distribution'] = alloc_domains.to_dict()
        print(f"\n🏷️  Domain Distribution (Top 5):")
        for domain, count in alloc_domains.head(5).items():
            print(f"   {domain}: {count} allocations")
        
        # 6. One-to-One Constraint Check
        student_allocations = self.allocations_df['student_id'].value_counts()
        multiple_allocations = (student_allocations > 1).sum()
        metrics['students_with_multiple_allocations'] = int(multiple_allocations)
        if multiple_allocations > 0:
            print(f"\n⚠️  Warning: {multiple_allocations} students have multiple allocations!")
        else:
            print(f"\n✓ All students have at most one allocation (constraint satisfied)")
        
        self.results['allocation_performance'] = metrics
        return metrics
    
    def test_api_performance(self, num_requests: int = 50) -> Dict:
        """Test API response times and throughput"""
        print("\n" + "="*60)
        print("TESTING API PERFORMANCE")
        print("="*60)
        
        try:
            # Test health endpoint
            start = time.time()
            response = requests.get(f"{self.api_url}/", timeout=5)
            health_time = time.time() - start
            print(f"\n🏥 Health Check: {health_time*1000:.2f}ms")
            
            # Get sample students
            response = requests.get(f"{self.api_url}/students", timeout=10)
            if response.status_code != 200:
                print("⚠️  Could not fetch students. Is the API running?")
                return {}
            
            students_data = response.json()
            student_ids = students_data.get('students', [])[:num_requests]
            
            if not student_ids:
                print("⚠️  No students found in API")
                return {}
            
            # Test recommendation endpoint
            response_times = []
            success_count = 0
            
            print(f"\n🔄 Testing {len(student_ids)} recommendation requests...")
            for i, student_id in enumerate(student_ids):
                start = time.time()
                try:
                    response = requests.get(
                        f"{self.api_url}/recommend/{student_id}?top_k=10",
                        timeout=10
                    )
                    elapsed = time.time() - start
                    response_times.append(elapsed)
                    if response.status_code == 200:
                        success_count += 1
                except Exception as e:
                    print(f"   Error for {student_id}: {e}")
                
                if (i + 1) % 10 == 0:
                    print(f"   Processed {i + 1}/{len(student_ids)} requests...")
            
            metrics = {
                'health_check_time_ms': health_time * 1000,
                'avg_response_time_ms': np.mean(response_times) * 1000,
                'median_response_time_ms': np.median(response_times) * 1000,
                'p95_response_time_ms': np.percentile(response_times, 95) * 1000,
                'p99_response_time_ms': np.percentile(response_times, 99) * 1000,
                'min_response_time_ms': np.min(response_times) * 1000,
                'max_response_time_ms': np.max(response_times) * 1000,
                'throughput_rps': len(response_times) / sum(response_times) if response_times else 0,
                'success_rate': success_count / len(student_ids) if student_ids else 0,
                'total_requests': len(student_ids)
            }
            
            print(f"\n⚡ API Performance Metrics:")
            print(f"   Average Response Time: {metrics['avg_response_time_ms']:.2f}ms")
            print(f"   Median Response Time: {metrics['median_response_time_ms']:.2f}ms")
            print(f"   P95 Response Time: {metrics['p95_response_time_ms']:.2f}ms")
            print(f"   P99 Response Time: {metrics['p99_response_time_ms']:.2f}ms")
            print(f"   Throughput: {metrics['throughput_rps']:.2f} requests/second")
            print(f"   Success Rate: {metrics['success_rate']*100:.2f}%")
            
            self.results['api_performance'] = metrics
            return metrics
            
        except requests.exceptions.ConnectionError:
            print("⚠️  Could not connect to API. Make sure it's running at", self.api_url)
            return {}
        except Exception as e:
            print(f"⚠️  Error testing API: {e}")
            return {}
    
    def test_recommendation_consistency(self) -> Dict:
        """Test consistency of recommendations"""
        print("\n" + "="*60)
        print("TESTING RECOMMENDATION CONSISTENCY")
        print("="*60)
        
        metrics = {}
        
        # Check if recommendations are properly ranked
        consistency_errors = 0
        for student_id in self.recommendations_df['student_id'].unique()[:50]:
            student_recs = self.recommendations_df[
                self.recommendations_df['student_id'] == student_id
            ].sort_values('rank')
            
            # Check if ranks are sequential
            ranks = student_recs['rank'].values
            if not np.array_equal(ranks, np.arange(1, len(ranks) + 1)):
                consistency_errors += 1
            
            # Check if scores are descending
            scores = student_recs['score'].values
            if not np.all(scores[:-1] >= scores[1:]):
                consistency_errors += 1
        
        metrics['consistency_errors'] = consistency_errors
        metrics['consistency_rate'] = 1.0 - (consistency_errors / 50) if consistency_errors < 50 else 0.0
        
        print(f"\n✓ Consistency Check:")
        print(f"   Errors Found: {consistency_errors}/50")
        print(f"   Consistency Rate: {metrics['consistency_rate']*100:.2f}%")
        
        self.results['consistency'] = metrics
        return metrics
    
    def _calculate_gini(self, values: np.ndarray) -> float:
        """Calculate Gini coefficient for popularity distribution"""
        if len(values) == 0:
            return 0.0
        sorted_values = np.sort(values)
        n = len(sorted_values)
        index = np.arange(1, n + 1)
        return (2 * np.sum(index * sorted_values)) / (n * np.sum(sorted_values)) - (n + 1) / n
    
    def generate_report(self):
        """Generate comprehensive performance report"""
        print("\n" + "="*60)
        print("GENERATING PERFORMANCE REPORT")
        print("="*60)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = RESULTS_DIR / f"performance_report_{timestamp}.json"
        
        # Add summary statistics
        summary = {
            'timestamp': timestamp,
            'total_students': len(self.students_df),
            'total_internships': len(self.internships_df),
            'total_recommendations': len(self.recommendations_df),
            'total_allocations': len(self.allocations_df),
            'results': self.results
        }
        
        # Save JSON report
        with open(report_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        print(f"\n✓ Report saved to: {report_file}")
        
        # Generate markdown summary
        md_file = RESULTS_DIR / f"performance_summary_{timestamp}.md"
        with open(md_file, 'w') as f:
            f.write("# Performance Test Report\n\n")
            f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write("## Summary\n\n")
            f.write(f"- Total Students: {len(self.students_df)}\n")
            f.write(f"- Total Internships: {len(self.internships_df)}\n")
            f.write(f"- Total Recommendations: {len(self.recommendations_df)}\n")
            f.write(f"- Total Allocations: {len(self.allocations_df)}\n\n")
            
            if 'recommendation_quality' in self.results:
                f.write("## Recommendation Quality\n\n")
                rq = self.results['recommendation_quality']
                f.write(f"- Mean Score: {rq.get('score_mean', 0):.4f}\n")
                f.write(f"- Student Coverage: {rq.get('student_coverage', 0)*100:.2f}%\n")
                f.write(f"- Internship Coverage: {rq.get('internship_coverage', 0)*100:.2f}%\n")
                f.write(f"- Diversity: {rq.get('diversity_mean', 0):.4f}\n\n")
            
            if 'allocation_performance' in self.results:
                f.write("## Allocation Performance\n\n")
                ap = self.results['allocation_performance']
                f.write(f"- Utilization Rate: {ap.get('utilization_rate', 0)*100:.2f}%\n")
                f.write(f"- Average Allocation Score: {ap.get('alloc_score_mean', 0):.4f}\n\n")
            
            if 'api_performance' in self.results:
                f.write("## API Performance\n\n")
                api = self.results['api_performance']
                f.write(f"- Average Response Time: {api.get('avg_response_time_ms', 0):.2f}ms\n")
                f.write(f"- Throughput: {api.get('throughput_rps', 0):.2f} req/s\n")
                f.write(f"- Success Rate: {api.get('success_rate', 0)*100:.2f}%\n\n")
        
        print(f"✓ Markdown summary saved to: {md_file}")
        
        return report_file, md_file
    
    def run_all_tests(self, test_api: bool = True):
        """Run all performance tests"""
        print("="*60)
        print("HYBRID RECOMMENDER SYSTEM - PERFORMANCE TESTING")
        print("="*60)
        
        self.load_data()
        self.test_recommendation_quality()
        self.test_allocation_performance()
        self.test_recommendation_consistency()
        
        if test_api:
            self.test_api_performance()
        
        self.generate_report()
        
        print("\n" + "="*60)
        print("✅ ALL TESTS COMPLETED")
        print("="*60)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test recommender system performance")
    parser.add_argument("--api-url", default=API_BASE_URL, help="API base URL")
    parser.add_argument("--skip-api", action="store_true", help="Skip API performance tests")
    parser.add_argument("--out-dir", default=OUT_DIR, type=Path, help="Output directory for data files")
    
    args = parser.parse_args()
    
    tester = PerformanceTester(api_url=args.api_url, out_dir=args.out_dir)
    tester.run_all_tests(test_api=not args.skip_api)

