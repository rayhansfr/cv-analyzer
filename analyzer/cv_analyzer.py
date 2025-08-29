#!/usr/bin/env python3
"""
Simple CV Analyzer - Main Entry Point
Uses Qwen2.5:1.5b-instruct for CV analysis, reasoning, and ranking
"""

import argparse
import sys
import os
import json
from pathlib import Path

# Add processors to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'processors'))

from simple_cv_analyzer import SimpleCVAnalyzer

def main():
    parser = argparse.ArgumentParser(description='CV Analyzer with AI Reasoning')
    parser.add_argument('--file', '-f', type=str, help='Single PDF file to analyze')
    parser.add_argument('--folder', '-d', type=str, help='Folder containing PDF files to analyze')
    parser.add_argument('--output', '-o', type=str, help='Output JSON file (optional)')
    parser.add_argument('--detailed', action='store_true', help='Show detailed output')
    
    args = parser.parse_args()
    
    if not args.file and not args.folder:
        print("❌ Please provide either a file (--file) or folder (--folder) to analyze")
        parser.print_help()
        return
    
    # Initialize analyzer
    try:
        analyzer = SimpleCVAnalyzer()
    except Exception as e:
        print(f"❌ Failed to initialize CV analyzer: {e}")
        return
    
    if args.file:
        # Analyze single file
        if not os.path.exists(args.file):
            print(f"❌ File not found: {args.file}")
            return
        
        print(f"\\n🚀 Analyzing single CV: {args.file}")
        result = analyzer.analyze_cv(args.file)
        
        if args.detailed:
            print_detailed_result(result)
        else:
            print_summary_result(result)
            
        if args.output:
            save_results([result], args.output)
    
    elif args.folder:
        # Analyze multiple files
        if not os.path.exists(args.folder):
            print(f"❌ Folder not found: {args.folder}")
            return
        
        print(f"\\n🚀 Analyzing CVs in folder: {args.folder}")
        results = analyzer.analyze_multiple_cvs(args.folder, args.output)
        
        if results:
            print_batch_summary(results)
            if args.detailed:
                for result in results[:5]:  # Show top 5
                    print_detailed_result(result)

def print_detailed_result(result):
    """Print detailed analysis result"""
    print(f"\\n{'='*80}")
    print(f"📄 File: {result['filename']}")
    print(f"{'='*80}")
    
    if result['status'] == 'error':
        print(f"❌ Error: {result['error']}")
        return
    
    print(f"\\n📋 EXTRACTED INFORMATION:")
    print(result['extracted_info'])
    
    print(f"\\n📝 CANDIDATE SUMMARY:")
    print(result['candidate_summary'])
    
    print(f"\\n🤔 JOB FIT ANALYSIS:")
    fit_analysis = result['job_fit_analysis']
    print(f"Recommendation: {fit_analysis['recommendation']}")
    print(f"Fit Score: {fit_analysis['fit_score']:.1%}")
    print(f"\\nReasoning:")
    print(fit_analysis['reasoning'])
    
    print(f"\\n⏱️ Processing Time: {result['processing_time_seconds']:.2f} seconds")

def print_summary_result(result):
    """Print summary result"""
    if result['status'] == 'error':
        print(f"❌ {result['filename']}: {result['error']}")
        return
    
    fit_analysis = result['job_fit_analysis']
    print(f"\\n✅ {result['filename']}")
    print(f"   📊 Fit Score: {fit_analysis['fit_score']:.1%}")
    print(f"   🎯 Recommendation: {fit_analysis['recommendation']}")
    print(f"   ⏱️ Time: {result['processing_time_seconds']:.1f}s")

def print_batch_summary(results):
    """Print summary for batch processing"""
    successful = [r for r in results if r['status'] == 'success']
    failed = [r for r in results if r['status'] == 'error']
    
    print(f"\\n{'='*60}")
    print(f"📊 BATCH ANALYSIS SUMMARY")
    print(f"{'='*60}")
    print(f"Total files: {len(results)}")
    print(f"Successful: {len(successful)}")
    print(f"Failed: {len(failed)}")
    
    if successful:
        # Sort by fit score
        successful.sort(key=lambda x: x['job_fit_analysis']['fit_score'], reverse=True)
        
        print(f"\\n🏆 TOP CANDIDATES:")
        for i, result in enumerate(successful[:10], 1):
            fit_score = result['job_fit_analysis']['fit_score']
            recommendation = result['job_fit_analysis']['recommendation']
            print(f"{i:2d}. {result['filename']:<30} {fit_score:.1%} ({recommendation})")

def save_results(results, output_file):
    """Save results to JSON file"""
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\\n💾 Results saved to: {output_file}")
    except Exception as e:
        print(f"❌ Failed to save results: {e}")

if __name__ == "__main__":
    main()
