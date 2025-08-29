#!/usr/bin/env python3
"""
Simple CV Analyzer with Qwen2.5:1.5b-instruct
Extracts CV info, generates summary, and reasons about job fit
"""

import os
import json
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

# PDF processing imports
from pdfminer.high_level import extract_text
from pdfminer.layout import LAParams

# Hugging Face imports
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

class SimpleCVAnalyzer:
    def __init__(self):
        """Initialize the CV analyzer with Qwen2.5 model"""
        self.model_name = "Qwen/Qwen2.5-1.5B-Instruct"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Job requirements (you can modify this)
        self.job_requirements = {
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
        }
        
        print("Initializing Qwen2.5 model...")
        self._initialize_model()
        
    def _initialize_model(self):
        """Initialize Qwen2.5 model and tokenizer"""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True
            )
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None,
                trust_remote_code=True
            )
            
            print(f"✓ Qwen2.5 model loaded successfully on {self.device}")
            
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
    
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text from PDF file using pdfminer.six"""
        text = ""
        try:
            # Configure layout analysis parameters for better text extraction
            laparams = LAParams(
                boxes_flow=0.5,     # Flow threshold for combining text boxes
                word_margin=0.1,    # Word margin for combining characters
                char_margin=2.0,    # Character margin for combining characters
                line_margin=0.5,    # Line margin for combining text lines
                detect_vertical=False  # Disable vertical text detection for speed
            )
            
            # Extract text with improved layout analysis
            text = extract_text(pdf_path, laparams=laparams)
            
            # Clean up the extracted text
            text = self._clean_extracted_text(text)
            
        except Exception as e:
            print(f"Error reading PDF {pdf_path}: {e}")
            # Fallback: try without layout parameters
            try:
                text = extract_text(pdf_path)
                text = self._clean_extracted_text(text)
            except Exception as fallback_e:
                print(f"Fallback extraction also failed: {fallback_e}")
                text = ""
        
        return text.strip()
    
    def _clean_extracted_text(self, text: str) -> str:
        """Clean and normalize extracted text"""
        if not text:
            return ""
        
        # Remove excessive whitespace and normalize line breaks
        import re
        
        # Replace multiple spaces with single space
        text = re.sub(r' +', ' ', text)
        
        # Replace multiple newlines with single newline
        text = re.sub(r'\n+', '\n', text)
        
        # Remove leading/trailing whitespace from each line
        lines = [line.strip() for line in text.split('\n')]
        
        # Remove empty lines
        lines = [line for line in lines if line]
        
        # Join lines back
        text = '\n'.join(lines)
        
        return text
    
    def generate_response(self, prompt: str, max_length: int = 1000) -> str:
        """Generate response using Qwen2.5 model"""
        try:
            # Format the prompt for chat
            messages = [
                {"role": "system", "content": "You are a professional HR assistant that analyzes CVs with strict accuracy. You MUST only extract and use information that is explicitly stated in the provided documents. Never assume, infer, or add information that is not clearly written. If information is not present, you must state that it is 'not mentioned' or 'not provided'. Be conservative and factual in all responses."},
                {"role": "user", "content": prompt}
            ]
            
            # Apply chat template
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            
            # Tokenize
            model_inputs = self.tokenizer([text], return_tensors="pt").to(self.device)
            
            # Generate with more conservative parameters to reduce hallucination
            with torch.no_grad():
                generated_ids = self.model.generate(
                    model_inputs.input_ids,
                    max_new_tokens=max_length,
                    do_sample=True,
                    temperature=0.3,  # Lower temperature for more conservative responses
                    top_p=0.6,        # More focused sampling
                    repetition_penalty=1.1,  # Reduce repetition
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode response
            generated_ids = [
                output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]
            
            response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
            return response.strip()
            
        except Exception as e:
            print(f"Error generating response: {e}")
            return f"Error: {str(e)}"
    
    def extract_cv_information(self, cv_text: str) -> Dict:
        """Extract structured information from CV using AI"""
        prompt = f"""
        IMPORTANT: Only extract information that is EXPLICITLY mentioned in the CV text below. Do NOT add, assume, or infer any information that is not clearly stated in the document.

        CV Text:
        {cv_text[:3000]}  # Limit text length

        Extract ONLY the following information that is clearly mentioned in the CV:
        1. Full Name (if clearly stated)
        2. Contact Information (only email, phone if present)
        3. Education (only degrees, universities, years that are explicitly mentioned)
        4. Work Experience (only companies, positions, years that are clearly stated)
        5. Technical Skills (only skills explicitly listed or mentioned)
        6. Soft Skills (only if explicitly mentioned)
        7. Years of Total Experience (only if clearly calculable from dates provided)
        8. Professional Summary/Objective (only if present in CV)

        CRITICAL RULES:
        - If information is not in the CV, write "Not mentioned" or "Not provided"
        - Do not assume or add skills that are not explicitly listed
        - Do not infer experience with technologies not mentioned
        - Stick strictly to what is written in the document
        - Use exact words and phrases from the CV when possible

        Format your response clearly with labeled sections.
        """
        
        print("🔍 Extracting CV information...")
        return self.generate_response(prompt, max_length=800)
    
    def generate_candidate_summary(self, cv_info: str) -> str:
        """Generate a professional summary of the candidate"""
        prompt = f"""
        IMPORTANT: Base your summary ONLY on the information provided below. Do NOT add or assume any skills, experience, or qualifications that are not explicitly mentioned.

        CV Information:
        {cv_info}

        Create a concise professional summary (3-4 sentences) using ONLY the information above.
        
        Focus ONLY on:
        - Their actual area of expertise (as stated)
        - Actual years of experience (if mentioned)
        - Only the technical skills explicitly listed
        - Only achievements or qualifications actually mentioned

        CRITICAL RULES:
        - Do not mention any technology, skill, or experience not explicitly stated
        - Do not assume or infer additional capabilities
        - If certain information is missing, do not fill in gaps
        - Use only facts from the provided CV information

        Keep it concise and strictly factual.
        """
        
        print("📝 Generating candidate summary...")
        return self.generate_response(prompt, max_length=300)
    
    def reason_job_fit(self, cv_info: str, candidate_summary: str) -> Dict:
        """Analyze why the candidate fits or doesn't fit the job requirements"""
        job_req_text = f"""
        Position: {self.job_requirements['title']}
        Required Skills: {', '.join(self.job_requirements['requirements']['skills'])}
        Experience Required: {self.job_requirements['requirements']['experience_years']} years
        Education: {self.job_requirements['requirements']['education']}
        Domain Knowledge: {', '.join(self.job_requirements['requirements']['domain_knowledge'])}
        Preferred Skills: {', '.join(self.job_requirements['requirements']['preferred'])}
        """
        
        prompt = f"""
        CRITICAL INSTRUCTION: Analyze this candidate's fit based ONLY on information explicitly mentioned in their CV. Do NOT assume or add any skills, experience, or qualifications not stated in the candidate information.

        JOB REQUIREMENTS:
        {job_req_text}

        CANDIDATE INFORMATION (extracted from CV):
        {cv_info}

        CANDIDATE SUMMARY:
        {candidate_summary}

        Provide analysis using ONLY the facts from the candidate's CV:

        1. STRENGTHS (Only what's explicitly mentioned):
        - Skills they actually have (only those listed in CV)
        - Experience they actually have (only what's documented)
        - Qualifications they actually possess

        2. GAPS/WEAKNESSES (Based on what's missing from CV):
        - Required skills not mentioned in their CV
        - Experience gaps based on what's documented
        - Missing qualifications not listed in CV

        3. OVERALL RECOMMENDATION:
        - Would you recommend this candidate? (Yes/No/Maybe)
        - Base reasoning only on documented skills and experience
        - Fit percentage (0-100%) based on actual matches
        - Recommend rule: (Yes > 60% fit) (Maybe 40% to 60% fit) (No < 40% fit)

        4. SPECIFIC REASONING:
        - How do their documented technical skills match requirements?
        - Is their documented experience level appropriate?
        - Does their stated education align?

        CRITICAL RULES:
        - Only consider skills explicitly mentioned in the CV
        - Do not assume they have skills similar to what they have
        - Do not infer capabilities from job titles alone
        - Be strictly factual and conservative in assessment
        - If a skill is not mentioned, consider it as "not demonstrated"

        Be honest and strictly based on documented evidence only.
        """
        
        print("🤔 Analyzing job fit...")
        reasoning = self.generate_response(prompt, max_length=1000)
        
        # Try to extract a numerical score from the reasoning
        score = self._extract_fit_score(reasoning)
        
        return {
            'reasoning': reasoning,
            'fit_score': score,
            'recommendation': self._extract_recommendation(reasoning)
        }
    
    def _extract_fit_score(self, reasoning: str) -> float:
        """Extract numerical fit score from reasoning text"""
        import re
        # Look for percentage patterns
        patterns = [
            r'(\d+)%',
            r'(\d+)/100',
            r'score.*?(\d+)',
            r'fit.*?(\d+)'
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, reasoning.lower())
            if matches:
                try:
                    score = int(matches[0])
                    return min(100, max(0, score)) / 100.0
                except ValueError:
                    continue
        
        # Default scoring based on keywords
        if 'excellent' in reasoning.lower() or 'perfect' in reasoning.lower():
            return 0.9
        elif 'good' in reasoning.lower() or 'suitable' in reasoning.lower():
            return 0.7
        elif 'adequate' in reasoning.lower() or 'average' in reasoning.lower():
            return 0.6
        elif 'poor' in reasoning.lower() or 'not suitable' in reasoning.lower():
            return 0.3
        else:
            return 0.5
    
    def _extract_recommendation(self, reasoning: str) -> str:
        """Extract recommendation from reasoning text"""
        reasoning_lower = reasoning.lower()
        if 'yes' in reasoning_lower and 'recommend' in reasoning_lower:
            return 'Yes'
        elif 'no' in reasoning_lower and 'recommend' in reasoning_lower:
            return 'No'
        elif 'maybe' in reasoning_lower:
            return 'Maybe'
        else:
            return 'Maybe'
    
    def analyze_cv(self, pdf_path: str) -> Dict:
        """Complete CV analysis pipeline"""
        start_time = datetime.now()
        
        print(f"\n{'='*60}")
        print(f"📄 Analyzing CV: {os.path.basename(pdf_path)}")
        print(f"{'='*60}")
        
        result = {
            'filename': os.path.basename(pdf_path),
            'file_path': str(pdf_path),
            'analysis_timestamp': start_time.isoformat(),
            'status': 'success'
        }
        
        try:
            # Step 1: Extract text from PDF
            print("📖 Extracting text from PDF...")
            cv_text = self.extract_text_from_pdf(pdf_path)
            
            if not cv_text or len(cv_text.strip()) < 100:
                result['status'] = 'error'
                result['error'] = 'Could not extract sufficient text from PDF'
                return result
            
            # Step 2: Extract structured information
            cv_info = self.extract_cv_information(cv_text)
            result['extracted_info'] = cv_info
            
            # Step 3: Generate candidate summary
            candidate_summary = self.generate_candidate_summary(cv_info)
            result['candidate_summary'] = candidate_summary
            
            # Step 4: Analyze job fit
            job_fit_analysis = self.reason_job_fit(cv_info, candidate_summary)
            result['job_fit_analysis'] = job_fit_analysis
            
            # Step 5: Calculate processing time
            end_time = datetime.now()
            result['processing_time_seconds'] = (end_time - start_time).total_seconds()
            
            print(f"\n✅ Analysis completed in {result['processing_time_seconds']:.2f} seconds")
            
        except Exception as e:
            result['status'] = 'error'
            result['error'] = str(e)
            print(f"❌ Error analyzing CV: {e}")
        
        return result
    
    def analyze_multiple_cvs(self, cv_folder: str, output_file: str = None) -> List[Dict]:
        """Analyze multiple CVs and rank them"""
        if output_file is None:
            output_file = f"cv_analysis_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        cv_files = list(Path(cv_folder).glob("*.pdf"))
        print(f"\n🚀 Found {len(cv_files)} PDF files to analyze")
        
        if not cv_files:
            print("❌ No PDF files found in the specified folder")
            return []
        
        results = []
        start_time = datetime.now()
        
        # Analyze each CV
        for i, cv_file in enumerate(cv_files, 1):
            print(f"\n[{i}/{len(cv_files)}] Processing: {cv_file.name}")
            result = self.analyze_cv(str(cv_file))
            results.append(result)
        
        # Sort by fit score (highest first)
        successful_results = [r for r in results if r['status'] == 'success']
        if successful_results:
            successful_results.sort(
                key=lambda x: x.get('job_fit_analysis', {}).get('fit_score', 0), 
                reverse=True
            )
        
        # Create final output
        total_time = (datetime.now() - start_time).total_seconds()
        final_output = {
            'analysis_summary': {
                'total_cvs': len(cv_files),
                'successful_analyses': len(successful_results),
                'failed_analyses': len(results) - len(successful_results),
                'total_processing_time': total_time,
                'average_time_per_cv': total_time / len(cv_files) if cv_files else 0,
                'analysis_timestamp': datetime.now().isoformat()
            },
            'job_requirements': self.job_requirements,
            'results': results,
            'ranked_candidates': successful_results
        }
        
        # Save results
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(final_output, f, indent=2, ensure_ascii=False)
        
        # Print summary
        print(f"\n{'='*80}")
        print(f"📊 ANALYSIS SUMMARY")
        print(f"{'='*80}")
        print(f"Total CVs processed: {len(cv_files)}")
        print(f"Successful analyses: {len(successful_results)}")
        print(f"Total time: {total_time:.2f} seconds")
        print(f"Average per CV: {total_time/len(cv_files):.2f} seconds")
        print(f"Results saved to: {output_file}")
        
        if successful_results:
            print(f"\n🏆 TOP 5 CANDIDATES:")
            for i, result in enumerate(successful_results[:5], 1):
                fit_score = result.get('job_fit_analysis', {}).get('fit_score', 0)
                recommendation = result.get('job_fit_analysis', {}).get('recommendation', 'Unknown')
                print(f"{i}. {result['filename']} - Fit Score: {fit_score:.1%} - Recommendation: {recommendation}")
        
        return results
    
    def print_detailed_analysis(self, result: Dict):
        """Print detailed analysis for a single CV"""
        if result['status'] != 'success':
            print(f"❌ Analysis failed: {result.get('error', 'Unknown error')}")
            return
        
        print(f"\n{'='*80}")
        print(f"📄 DETAILED ANALYSIS: {result['filename']}")
        print(f"{'='*80}")
        
        print(f"\n📝 CANDIDATE SUMMARY:")
        print(result.get('candidate_summary', 'Not available'))
        
        print(f"\n🔍 EXTRACTED INFORMATION:")
        print(result.get('extracted_info', 'Not available'))
        
        fit_analysis = result.get('job_fit_analysis', {})
        print(f"\n🤔 JOB FIT ANALYSIS:")
        print(f"Fit Score: {fit_analysis.get('fit_score', 0):.1%}")
        print(f"Recommendation: {fit_analysis.get('recommendation', 'Unknown')}")
        print(f"\nDetailed Reasoning:")
        print(fit_analysis.get('reasoning', 'Not available'))
        
        print(f"\n⏱️ Processing Time: {result.get('processing_time_seconds', 0):.2f} seconds")


def main():
    """Main function to run the CV analyzer"""
    try:
        # Initialize analyzer
        analyzer = SimpleCVAnalyzer()
        
        # Set CV folder path
        cv_folder = "../cv"
        
        if not os.path.exists(cv_folder):
            print(f"❌ CV folder '{cv_folder}' not found!")
            print("Please create the folder and add PDF files to analyze.")
            return
        
        # Analyze all CVs
        results = analyzer.analyze_multiple_cvs(cv_folder)
        
        # Show detailed analysis for top candidate
        successful_results = [r for r in results if r['status'] == 'success']
        if successful_results:
            print(f"\n{'='*80}")
            print(f"🥇 DETAILED ANALYSIS FOR TOP CANDIDATE")
            analyzer.print_detailed_analysis(successful_results[0])
        
    except Exception as e:
        print(f"❌ Error in main: {e}")


if __name__ == "__main__":
    main()
