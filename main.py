#!/usr/bin/env python3
"""
CV Analyzer - Main Entry Point
Interactive CV analysis system with job description setup
"""

import os
import sys
import json
from pathlib import Path
from datetime import datetime

try:
    from analyzer.processors.simple_cv_analyzer import SimpleCVAnalyzer
except ImportError:
    print("❌ Error: Could not import SimpleCVAnalyzer")
    print("Make sure you're running from the correct directory and have installed dependencies")
    sys.exit(1)

class CVAnalysisMain:
    def __init__(self):
        self.analyzer = None
        self.job_templates = {
            'data_analyst': {
                'title': 'Senior Data Analyst',
                'department': 'Analytics & Business Intelligence',
                'requirements': {
                    'education': 'Bachelor degree in Statistics, Mathematics, Computer Science, or related field',
                    'skills': ['Python', 'SQL', 'Data Analysis', 'Excel', 'Tableau', 'Power BI', 'Statistics'],
                    'experience_years': 3,
                    'domain_knowledge': ['Business Intelligence', 'Data Visualization', 'Statistical Analysis'],
                    'soft_skills': ['Communication', 'Problem Solving', 'Team Collaboration'],
                    'preferred': ['Machine Learning', 'R', 'Advanced Excel', 'Database Management']
                }
            },
            'software_engineer': {
                'title': 'Software Engineer',
                'department': 'Engineering',
                'requirements': {
                    'education': 'Bachelor degree in Computer Science, Software Engineering, or related field',
                    'skills': ['Python', 'JavaScript', 'SQL', 'Git', 'REST APIs', 'Database Design'],
                    'experience_years': 2,
                    'domain_knowledge': ['Web Development', 'Software Architecture', 'Database Management'],
                    'soft_skills': ['Problem Solving', 'Team Collaboration', 'Communication'],
                    'preferred': ['React', 'Node.js', 'Docker', 'AWS', 'CI/CD']
                }
            },
            'project_manager': {
                'title': 'Project Manager',
                'department': 'Operations',
                'requirements': {
                    'education': 'Bachelor degree in Business, Management, or related field',
                    'skills': ['Project Management', 'Agile', 'Scrum', 'Risk Management', 'Budget Planning'],
                    'experience_years': 5,
                    'domain_knowledge': ['Project Lifecycle', 'Stakeholder Management', 'Process Improvement'],
                    'soft_skills': ['Leadership', 'Communication', 'Negotiation', 'Time Management'],
                    'preferred': ['PMP Certification', 'JIRA', 'MS Project', 'Six Sigma']
                }
            }
        }
    
    def display_banner(self):
        print("\n" + "="*70)
        print("🎯 CV ANALYZER - AI-Powered Recruitment Assistant")
        print("="*70)
        print("📋 Features:")
        print("   • CV text extraction and analysis")
        print("   • AI-powered candidate summaries")
        print("   • Job fit reasoning and scoring")
        print("   • Anti-hallucination technology")
        print("="*70)
    
    def select_job_template(self):
        """Let user select or create job description"""
        print("\n🏢 JOB DESCRIPTION SETUP")
        print("-" * 40)
        print("Choose an option:")
        print("1. Use predefined template")
        print("2. Create custom job description")
        print("3. Load from file")
        
        while True:
            choice = input("\nEnter your choice (1-3): ").strip()
            
            if choice == '1':
                return self.select_predefined_template()
            elif choice == '2':
                return self.create_custom_job()
            elif choice == '3':
                return self.load_job_from_file()
            else:
                print("❌ Invalid choice. Please enter 1, 2, or 3.")
    
    def select_predefined_template(self):
        """Select from predefined job templates"""
        print("\n📋 Available Templates:")
        templates = list(self.job_templates.keys())
        
        for i, template_key in enumerate(templates, 1):
            template = self.job_templates[template_key]
            print(f"{i}. {template['title']} - {template['department']}")
        
        while True:
            try:
                choice = int(input(f"\nSelect template (1-{len(templates)}): "))
                if 1 <= choice <= len(templates):
                    template_key = templates[choice - 1]
                    selected_template = self.job_templates[template_key].copy()
                    
                    print(f"\n✅ Selected: {selected_template['title']}")
                    self.display_job_requirements(selected_template)
                    
                    if input("\nUse this template? (y/n): ").lower().startswith('y'):
                        return selected_template
                    else:
                        continue
                else:
                    print(f"❌ Please enter a number between 1 and {len(templates)}")
            except ValueError:
                print("❌ Please enter a valid number")
    
    def create_custom_job(self):
        """Create custom job description interactively"""
        print("\n🛠️ Create Custom Job Description")
        print("-" * 40)
        
        job_desc = {
            'title': input("Job Title: ").strip(),
            'department': input("Department: ").strip(),
            'requirements': {}
        }
        
        print("\n📚 Education Requirements:")
        job_desc['requirements']['education'] = input("Education (e.g., Bachelor's in Computer Science): ").strip()
        
        print("\n💼 Experience:")
        while True:
            try:
                years = int(input("Required years of experience: "))
                job_desc['requirements']['experience_years'] = years
                break
            except ValueError:
                print("❌ Please enter a valid number")
        
        print("\n🔧 Technical Skills (comma-separated):")
        skills_input = input("Skills: ").strip()
        job_desc['requirements']['skills'] = [skill.strip() for skill in skills_input.split(',') if skill.strip()]
        
        print("\n🎓 Domain Knowledge (comma-separated):")
        domain_input = input("Domain areas: ").strip()
        job_desc['requirements']['domain_knowledge'] = [area.strip() for area in domain_input.split(',') if area.strip()]
        
        print("\n🤝 Soft Skills (comma-separated):")
        soft_input = input("Soft skills: ").strip()
        job_desc['requirements']['soft_skills'] = [skill.strip() for skill in soft_input.split(',') if skill.strip()]
        
        print("\n⭐ Preferred/Nice-to-have (comma-separated):")
        preferred_input = input("Preferred skills: ").strip()
        job_desc['requirements']['preferred'] = [skill.strip() for skill in preferred_input.split(',') if skill.strip()]
        
        print("\n📋 Custom Job Description Created:")
        self.display_job_requirements(job_desc)
        
        if input("\nSave this job description? (y/n): ").lower().startswith('y'):
            self.save_job_description(job_desc)
        
        return job_desc
    
    def load_job_from_file(self):
        """Load job description from JSON file"""
        print("\n📁 Load Job Description from File")
        print("-" * 40)
        
        file_path = input("Enter path to JSON file: ").strip()
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                job_desc = json.load(f)
            
            print(f"\n✅ Loaded job description from {file_path}")
            self.display_job_requirements(job_desc)
            
            return job_desc
            
        except FileNotFoundError:
            print(f"❌ File not found: {file_path}")
            return self.select_job_template()
        except json.JSONDecodeError:
            print(f"❌ Invalid JSON format in {file_path}")
            return self.select_job_template()
        except Exception as e:
            print(f"❌ Error loading file: {e}")
            return self.select_job_template()
    
    def display_job_requirements(self, job_desc):
        """Display job requirements in a formatted way"""
        print(f"\n📋 {job_desc['title']} - {job_desc.get('department', 'N/A')}")
        print("-" * 50)
        
        req = job_desc['requirements']
        print(f"🎓 Education: {req.get('education', 'Not specified')}")
        print(f"📅 Experience: {req.get('experience_years', 0)} years")
        print(f"🔧 Skills: {', '.join(req.get('skills', []))}")
        print(f"🎯 Domain: {', '.join(req.get('domain_knowledge', []))}")
        print(f"🤝 Soft Skills: {', '.join(req.get('soft_skills', []))}")
        print(f"⭐ Preferred: {', '.join(req.get('preferred', []))}")
    
    def save_job_description(self, job_desc):
        """Save job description to file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"job_description_{timestamp}.json"
        
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(job_desc, f, indent=2, ensure_ascii=False)
            print(f"💾 Job description saved to: {filename}")
        except Exception as e:
            print(f"❌ Error saving job description: {e}")
    
    def select_cv_source(self):
        """Select CV files to analyze"""
        print("\n📁 CV SELECTION")
        print("-" * 40)
        print("Choose CV source:")
        print("1. Single CV file")
        print("2. Folder with multiple CVs")
        print("3. Use default CV folder (cv\)")
        
        while True:
            choice = input("\nEnter your choice (1-3): ").strip()
            
            if choice == '1':
                file_path = input("Enter path to CV file (.pdf): ").strip()
                if os.path.exists(file_path) and file_path.lower().endswith('.pdf'):
                    return 'single', file_path
                else:
                    print("❌ File not found or not a PDF. Please try again.")
                    
            elif choice == '2':
                folder_path = input("Enter path to CV folder: ").strip()
                if os.path.exists(folder_path) and os.path.isdir(folder_path):
                    pdf_files = list(Path(folder_path).glob("*.pdf"))
                    if pdf_files:
                        print(f"✅ Found {len(pdf_files)} PDF files")
                        return 'folder', folder_path
                    else:
                        print("❌ No PDF files found in folder. Please try again.")
                else:
                    print("❌ Folder not found. Please try again.")
                    
            elif choice == '3':
                default_path = "cv/"
                if os.path.exists(default_path):
                    pdf_files = list(Path(default_path).glob("*.pdf"))
                    if pdf_files:
                        print(f"✅ Found {len(pdf_files)} PDF files in default folder")
                        return 'folder', default_path
                    else:
                        print("❌ No PDF files found in default CV folder.")
                        continue
                else:
                    print("❌ Default CV folder not found.")
                    continue
            else:
                print("❌ Invalid choice. Please enter 1, 2, or 3.")
    
    def initialize_analyzer(self, job_requirements):
        """Initialize the CV analyzer with job requirements"""
        print("\n🤖 Initializing AI CV Analyzer...")
        print("⏳ Loading Qwen2.5 model (this may take a moment)...")
        
        try:
            self.analyzer = SimpleCVAnalyzer()
            # Update job requirements
            self.analyzer.job_requirements = job_requirements
            print("✅ CV Analyzer initialized successfully!")
            return True
        except Exception as e:
            print(f"❌ Failed to initialize analyzer: {e}")
            print("\n💡 Troubleshooting tips:")
            print("   • Check internet connection for model download")
            print("   • Ensure you have enough memory (4GB+ recommended)")
            print("   • Install requirements: pip install -r ai_cv_system/requirements_simple.txt")
            return False
    
    def run_analysis(self, source_type, source_path):
        """Run CV analysis"""
        print(f"\n🚀 Starting CV Analysis...")
        print("-" * 50)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if source_type == 'single':
            # Analyze single CV
            print(f"📄 Analyzing: {os.path.basename(source_path)}")
            result = self.analyzer.analyze_cv(source_path)
            
            self.display_single_result(result)
            
            # Save result
            output_file = f"cv_analysis_single_{timestamp}.json"
            self.save_results([result], output_file)
            
        else:
            # Analyze multiple CVs
            print(f"📁 Analyzing CVs in: {source_path}")
            results = self.analyzer.analyze_multiple_cvs(source_path)
            
            if results:
                self.display_batch_results(results)
                
                # Save results
                output_file = f"cv_analysis_batch_{timestamp}.json"
                self.save_results(results, output_file)
            else:
                print("❌ No CVs were successfully analyzed.")
    
    def display_single_result(self, result):
        """Display single CV analysis result"""
        print(f"\n{'='*60}")
        print(f"📄 ANALYSIS RESULT: {result['filename']}")
        print(f"{'='*60}")
        
        if result['status'] == 'error':
            print(f"❌ Error: {result['error']}")
            return
        
        # Job fit summary
        fit_analysis = result['job_fit_analysis']
        print(f"\n🎯 JOB FIT SUMMARY:")
        print(f"   Recommendation: {fit_analysis['recommendation']}")
        print(f"   Fit Score: {fit_analysis['fit_score']:.1%}")
        print(f"   Processing Time: {result['processing_time_seconds']:.2f}s")
        
        # Candidate summary
        print(f"\n📝 CANDIDATE SUMMARY:")
        print(f"   {result['candidate_summary']}")
        
        # Detailed reasoning
        print(f"\n🤔 AI REASONING:")
        reasoning_lines = fit_analysis['reasoning'].split('\n')
        for line in reasoning_lines:
            if line.strip():
                print(f"   {line.strip()}")
    
    def display_batch_results(self, results):
        """Display batch analysis results"""
        successful = [r for r in results if r['status'] == 'success']
        failed = [r for r in results if r['status'] == 'error']
        
        print(f"\n{'='*60}")
        print(f"📊 BATCH ANALYSIS SUMMARY")
        print(f"{'='*60}")
        print(f"Total files: {len(results)}")
        print(f"Successful: {len(successful)}")
        print(f"Failed: {len(failed)}")
        
        if successful:
            # Sort by fit score
            successful.sort(key=lambda x: x['job_fit_analysis']['fit_score'], reverse=True)
            
            print(f"\n🏆 TOP CANDIDATES (by fit score):")
            print("-" * 60)
            
            for i, result in enumerate(successful[:10], 1):
                fit_score = result['job_fit_analysis']['fit_score']
                recommendation = result['job_fit_analysis']['recommendation']
                filename = result['filename'][:35] + "..." if len(result['filename']) > 35 else result['filename']
                
                print(f"{i:2d}. {filename:<38} {fit_score:.1%} ({recommendation})")
            
            # Show detailed results for top 3
            print(f"\n📋 DETAILED ANALYSIS (Top 3):")
            print("=" * 60)
            
            for result in successful[:3]:
                self.display_single_result(result)
        
        if failed:
            print(f"\n❌ FAILED ANALYSES:")
            for result in failed:
                print(f"   • {result['filename']}: {result['error']}")
    
    def save_results(self, results, filename):
        """Save results to JSON file"""
        try:
            # Convert any Path objects to strings for JSON serialization
            clean_results = []
            for result in results:
                clean_result = result.copy()
                if 'file_path' in clean_result:
                    clean_result['file_path'] = str(clean_result['file_path'])
                clean_results.append(clean_result)
            
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump({
                    'analysis_timestamp': datetime.now().isoformat(),
                    'total_analyzed': len(results),
                    'successful': len([r for r in results if r['status'] == 'success']),
                    'results': clean_results
                }, f, indent=2, ensure_ascii=False)
            
            print(f"\n💾 Results saved to: {filename}")
            
        except Exception as e:
            print(f"❌ Error saving results: {e}")
    
    def run(self):
        """Main application loop"""
        self.display_banner()
        
        try:
            # Step 1: Setup job description
            job_requirements = self.select_job_template()
            
            # Step 2: Initialize analyzer
            if not self.initialize_analyzer(job_requirements):
                return
            
            # Step 3: Select CV source
            source_type, source_path = self.select_cv_source()
            
            # Step 4: Run analysis
            self.run_analysis(source_type, source_path)
            
            print(f"\n✅ Analysis completed successfully!")
            print(f"📁 Check the generated JSON file for detailed results.")
            
        except KeyboardInterrupt:
            print(f"\n\n⏹️ Analysis interrupted by user.")
        except Exception as e:
            print(f"\n❌ Unexpected error: {e}")
            print(f"Please check your setup and try again.")

def main():
    """Entry point"""
    app = CVAnalysisMain()
    app.run()

if __name__ == "__main__":
    main()