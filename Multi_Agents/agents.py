import modal
import re
import json

class Analyzer_Agent():
    def __init__(self, api_name, cls):
        self.agent_name = cls
        LLM = modal.Cls.from_name(api_name, cls)
        self.llm = LLM()
        self.system_prompt = """
            You are a CV-parsing assistant. A user will provide a CV in raw text format.
            Your goal is to:
                
            1. Parse the text and identify the following sections:
                - Name
                - Profile (or Summary/About Me)
                - Working Experience (or Work History/Professional Experience)
                - Education
                - Certifications
                - Skills
                - Projects
                
            2. Clean the extracted text (e.g., remove excessive whitespace or formatting artifacts).
                
            3. Return the information as a structured dictionary where:
                - Each section name is a key.
                - The corresponding value is a detailed description of that section.

            4. If any section is missing or cannot be identified, still include the key with an empty value.

            5. Do not include any additional commentary or text outside the dictionary.

            6. Do not answer with code or bullet points or anything else, just plain text.
                
            7. You **ALWAYS** answer in **DICTIONARY** format with only 7 keys and in English, the keys are the sections of the CV as shown in the examples below and the values a detailed summary with all the detail in paragraph.

            Example output:
                
                {
                "Name": "The name of the person"
                "Profile": "Short summary or introduction...",
                "Working Experience": "Work experience details in a paragraph...",
                "Education": "Educational background details in a paragraph...",
                "Certifications": "IBM Data Science, Tensorflow Developer,..."
                "Skills": "Paragraph of skills...",
                "Projects": "Description of relevant projects...",
                }

            Example output that not all the info are in the CV:
                
                {
                "Name": "The name of the person"
                "Profile": "Short summary or introduction...",
                "Working Experience": "",
                "Education": "Educational background details in a paragraph...",
                "Certifications": ""
                "Skills": "Paragraph of skills...",
                "Projects": "",
                }
        """

    def create_history(self, prompt: str) -> list:
        return [
            {'role': 'system', 'content': self.system_prompt},
            {'role': 'user', 'content': 'The CV that need to be parsed in JSON with 7 keys as format: \n' + prompt},
        ]

    def extract_json(self, response):    
        # Extract the JSON part from the first { to the last }
        json_match = re.search(r"\{.*\}", response, re.DOTALL)
        
        if json_match:
            return json_match.group(0)  # Return the JSON part
        else:
            return None  # No valid JSON found

    def generate_response(self, history: list) -> str:
        for _ in range(2):
            try:
                # Generate the response
                response = ''
                for chunk in self.llm.inference.remote_gen(history):
                    response += chunk

                response_json = self.extract_json(response) # Extract the JSON from the response
                json_obj = json.loads(response_json) # Parse it to create a JSON

                # Ensure correct keys exist in the parsed response
                if list(json_obj.keys()) == ['Name', 'Profile', 'Working Experience', 'Education', 'Certifications', 'Skills', 'Projects']:
                    return response_json
                else:
                    raise ValueError()  # Force re-processing if keys are incorrect
            except Exception as e:
                print(self.agent_name, ' Error: \n', e.args)
        return "This CV Analyzer Agent assistant couldn't parse the CV. Continue the judge without this one!!"

    def invoke(self, prompt: str) -> str:
        # Create the history
        history = self.create_history(prompt)
        
        # Generate response and return it
        response = self.generate_response(history)
        return response


class Judge_Agent():
    def __init__(self, api_name='mistral', cls='Mistral'):
        self.agent_name = cls
        LLM = modal.Cls.from_name(api_name, cls)
        self.llm = LLM()
        self.system_prompt = """
            You are an AI judge assistant that your job is to evaluate three CV analyses from different AI assistants.
            Your task is to **aggregate the most detailed and informative content** from each section across the three responses and construct the most comprehensive CV analysis.

            **Your criteria:**
                1. For each section:
                    - Identify which response provides the **most complete, clear, and detailed information**.
                    - If useful details exist in multiple responses, **combine them** into a single, well-structured description.
                    - Ensure **no information is lost** while avoiding redundant or repetitive phrasing.

                2. Prioritize:
                    - **Clarity, depth, and fluency.**
                    - **Richness of information** (e.g., more relevant experience, specific achievements, certifications, or skills).
                    - **Consistency in tone and structure** while ensuring natural readability.

                3. **Return a single structured dictionary** with the following sections:
                    - Name
                    - Profile
                    - Working Experience
                    - Education
                    - Certifications
                    - Skills
                    - Projects

                4. **Do not exclude any relevant details** the goal is to construct the **most enriched** CV analysis possible.

                5. Your response must be **in dictionary format only**. Do not add commentary, explanations, or formatting beyond the structured output.

            **Example Input:**


                CV Analyzer Assistant 1: 
                {
                    "Name": "John Doe",
                    "Profile": "Experienced software engineer specializing in AI and machine learning.",
                    "Working Experience": "Worked at Google for 5 years, leading AI research projects.",
                    "Education": "Master's in Computer Science from MIT.",
                    "Certifications": "TensorFlow Developer, AWS Certified Solutions Architect.",
                    "Skills": "Python, Deep Learning, NLP, Cloud Computing.",
                    "Projects": "Developed a state-of-the-art chatbot used by Fortune 500 companies."
                }

                CV Analyzer Assistant 2: 
                {
                    "Name": "John Doe",
                    "Profile": "Software engineer with 10+ years of experience in AI and ML.",
                    "Working Experience": "Led AI projects at Google and Facebook, specializing in NLP and deep learning. Designed scalable machine learning pipelines.",
                    "Education": "Master's in Computer Science from MIT, Bachelor's in Computer Engineering from Stanford.",
                    "Certifications": "TensorFlow Developer, AWS Certified, Google Cloud ML Engineer.",
                    "Skills": "Python, Deep Learning, NLP, Cloud Computing, MLOps, Data Engineering.",
                    "Projects": "Created an AI-driven recommendation system that improved user engagement by 40%."
                }

                CV Analyzer Assistant 3: 
                {
                    "Name": "John Doe",
                    "Profile": "AI expert with experience in software engineering.",
                    "Working Experience": "Worked at a tech company handling AI projects.",
                    "Education": "Master's in Computer Science.",
                    "Certifications": "TensorFlow Developer.",
                    "Skills": "Python, Machine Learning.",
                    "Projects": "Built an AI-powered chatbot."
                }

            **Example Output:**

                {
                    "Name": "John Doe",
                    "Profile": "Software engineer with over 10 years of experience in AI and ML. Specialized in natural language processing, deep learning, and scalable AI architectures. Led research and development efforts at Google and Facebook, designing innovative machine learning solutions.",
                    "Working Experience": "Worked at Google and Facebook, leading AI projects in NLP and deep learning. Designed and deployed scalable machine learning pipelines, improving system efficiency. Managed cross-functional teams and collaborated on AI-driven solutions that enhanced business processes.",
                    "Education": "Master's in Computer Science from MIT, Bachelor's in Computer Engineering from Stanford.",
                    "Certifications": "TensorFlow Developer, AWS Certified Solutions Architect, Google Cloud ML Engineer.",
                    "Skills": "Python, Deep Learning, NLP, Cloud Computing, MLOps, Data Engineering, Scalable AI Systems.",
                    "Projects": "Developed a state-of-the-art chatbot used by Fortune 500 companies. Created an AI-driven recommendation system that improved user engagement by 40%, contributing to increased customer retention. Led the development of AI-powered automation tools for enterprise applications."
                }
        """

    def create_history(self, analyzed_cvs: list) -> list:
        # Format the user prompt
        user_prompt = "Judge these CVs descriptions:\n\n"
        
        for i, cv in enumerate(analyzed_cvs, start=1):
            user_prompt += f"CV Analyzer Assistant {i}:\n{cv}\n\n"
        
        # Construct the history list
        return [
            {'role': 'system', 'content': self.system_prompt},
            {'role': 'user', 'content': user_prompt.strip()}  # Trim any trailing whitespace
        ]

    def extract_json(self, response):    
        # Extract the JSON part from the first { to the last }
        json_match = re.search(r"\{.*\}", response, re.DOTALL)
        
        if json_match:
            return json_match.group(0)  # Return the JSON part
        else:
            return None  # No valid JSON found

    def generate_response(self, history):
        for _ in range(3):
            try:
                # Generate the response
                response = ''
                for chunk in self.llm.inference.remote_gen(history):
                    response += chunk

                response_json = self.extract_json(response) # Extract the JSON from the response
                json_obj = json.loads(response_json) # Parse it to create a JSON

                # Ensure correct keys exist in the parsed response
                if list(json_obj.keys()) == ['Name', 'Profile', 'Working Experience', 'Education', 'Certifications', 'Skills', 'Projects']:
                    return response_json
                else:
                    raise ValueError()  # Force re-processing if keys are incorrect
            except Exception as e:
                print(self.agent_name, ' Error: \n', e.args)
        return None

    def invoke(self, prompts: list) -> str:
        # Create the history
        history = self.create_history(prompts)

        # Generate response and return it
        response = self.generate_response(history)
        return response