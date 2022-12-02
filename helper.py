# source: https://www.youtube.com/c/machinelearnear

import os
import sys
import shutil
import subprocess
import gradio as gr

from os.path import exists as path_exists
from typing import Dict

newline, bold, unbold = "\n", "\033[1m", "\033[0m"

class RepoHandler:
    def __init__(self, repo_url: str) -> None:
        """
        Initialize the RepoHandler class with the provided repository URL and requirements file.

        Args:
        - repo_url: URL of the git repository to clone.

        Returns:
        None
        """
        if 'google.colab' in str(get_ipython()):
            print(f"{bold}Running on Google Colab{unbold}")
        else:
            print(f"{bold}Running on SageMaker Studio Lab or locally{unbold}")
            
        self.repo_url = repo_url
        self.repo_name = self.repo_url.split('/')[-1]

    def __str__(self):
        if os.path.exists(self.repo_name):
            return f"{self.retrieve_readme(f'{self.repo_name}/README.md')}"
        else:
            print(f"{bold}The repo '{self.repo_name}' has not been cloned yet.{unbold}")
            
    def retrieve_readme(self, filename) -> Dict:
        readme = {}
        if path_exists(filename):
            with open(filename) as f:
                for line in f:
                    if not line.find(':') > 0 or 'Check' in line: continue
                    (k,v) = line.split(':')
                    readme[(k)] = v.strip().replace('\n','')
        else:
            print(f"{bold}No 'readme.md' file{unbold}")
            
        return readme
        
    def clone_repo(self, overwrite=False) -> None:
        """
        Clone the git repository specified in the repo_url attribute.

        Returns:
        None
        """
        # Check if repository has already been cloned locally
        if overwrite: 
            try:
                shutil.rmtree(self.repo_name)
            except OSError as e:
                print("Error: %s - %s." % (e.filename, e.strerror))
        if os.path.exists(self.repo_name):
            print(f"Repository {self.repo_name} has already been cloned.")
        else:
            # Clone the repository
            subprocess.run(["git", "clone", self.repo_url])

    def install_requirements(self, requirements_file: str = None, install_xformers: bool = False) -> None:
        """
        Install the requirements specified in the requirements_file attribute.
        
        Args:
        - requirements_file: Name of the file containing the requirements to install. This file must be 
        located in the root directory of the repository. Defaults to "requirements.txt".

        Returns:
        None
        """
        if not requirements_file: requirements_file = f"{self.repo_name}/{self.requirements_file}"
        
        # install requirements
        subprocess.run(["pip", "install", "-r", requirements_file])
        if install_xformers: self.install_xformers()

    def run_web_demo(self, aws_domain=None, aws_region=None) -> None:
        """
        Launch the Gradio or Streamlit web demo for the cloned repository.
        Works with Google Colab or SageMaker Studio Lab.

        Returns:
        None
        """
        import torch
        if torch.cuda.is_available(): self.get_gpu_memory_map()
        else: print(f"{bold}Not using the GPU{unbold}")
        
        readme = self.__str__()
        print(f"{bold}Demo: {readme['title']}{newline}{unbold}")
        print("Wait a few seconds, then click the link below to open your application:")
        if all([domain,region]):
              print(f'{bold}https://{domain}.studio.{region}.sagemaker.aws/studiolab/default/jupyter/proxy/6006/{unbold}')

        if readme["sdk"] == 'gradio':
            gr.close_all()
            os.system(f'export GRADIO_SERVER_PORT=6006 && cd {self.repo_name} && python {readme["app_file"]}')
        elif readme["title"] == 'streamlit':
            os.system(f'cd {self.repo_name} && streamlit run {readme["app_file"]} --server.port 6006')
        else:
            print('This notebook will not work with static apps hosted on "Spaces"')

    def get_gpu_memory_map(self) -> Dict[str, int]:
        """Get the current gpu usage.

        Return:
            A dictionary in which the keys are device ids as integers and
            values are memory usage as integers in MB.
        """
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,memory.free", "--format=csv,noheader",],
            encoding="utf-8",
            # capture_output=True,          # valid for python version >=3.7
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,  # for backward compatibility with python version 3.6
            check=True,
        )
        # Convert lines into a dictionary, return f"{}"
        gpu_memory = [x for x in result.stdout.strip().split(os.linesep)]
        gpu_memory_map = {f"{bold}gpu_{index}{unbold}": memory for index, memory in enumerate(gpu_memory)}
        return gpu_memory_map

    def install_xformers(self) -> None:
        from subprocess import getoutput
        from IPython.display import HTML
        from IPython.display import clear_output
        import time

        subprocess.run(["pip", "install", "-U", "--pre", "triton"])

        s = getoutput('nvidia-smi')
        if 'T4' in s: gpu = 'T4'
        elif 'P100' in s: gpu = 'P100'
        elif 'V100' in s: gpu = 'V100'
        elif 'A100' in s: gpu = 'A100'

        while True:
            try: 
                gpu=='T4'or gpu=='P100'or gpu=='V100'or gpu=='A100'
                break
            except:
                pass
            print(f'{bold} Seems that your GPU is not supported at the moment.{unbold}')
            time.sleep(5)

        if (gpu=='T4'): precompiled_wheels = "https://github.com/TheLastBen/fast-stable-diffusion/raw/main/precompiled/T4/xformers-0.0.13.dev0-py3-none-any.whl"
        elif (gpu=='P100'): precompiled_wheels = "https://github.com/TheLastBen/fast-stable-diffusion/raw/main/precompiled/P100/xformers-0.0.13.dev0-py3-none-any.whl"
        elif (gpu=='V100'): precompiled_wheels = "https://github.com/TheLastBen/fast-stable-diffusion/raw/main/precompiled/V100/xformers-0.0.13.dev0-py3-none-any.whl"
        elif (gpu=='A100'): precompiled_wheels = "https://github.com/TheLastBen/fast-stable-diffusion/raw/main/precompiled/A100/xformers-0.0.13.dev0-py3-none-any.whl"

        subprocess.run(["pip", "install", "-q", precompiled_wheels])