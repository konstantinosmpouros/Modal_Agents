import asyncio
from threading import Thread
import modal
from modal.exception import NotFoundError
import subprocess

# Configuration remains the same
APIS = {
    'llama': {'class_name': 'Llama', 'app_var': 'llama_app', 'app_name': 'llama-app'},
    'qwen': {'class_name': 'Qwen', 'app_var': 'qwen_app', 'app_name': 'qwen-app'},
    'phi': {'class_name': 'Phi', 'app_var': 'phi_app', 'app_name': 'phi-app'},
    'gemma': {'class_name': 'Gemma', 'app_var': 'gemma_app', 'app_name': 'gemma-app'},
    'mistral': {'class_name': 'Mistral', 'app_var': 'mistral_app', 'app_name': 'mistral-app'}
}

class APIManager:
    def __init__(self, apis: list[str]):
        # Check if the apis given to maintene are in those that we have developed. If not raise exception, else extract them from the APIS dictionary
        if invalid_apis := [api for api in apis if api not in APIS]:
            raise Exception(f"Invalid API(s): {', '.join(invalid_apis)}")
        self.apis_needed = {k: APIS[k] for k in apis}
        # Init an active apis dictionary to ping them and keep them warm every 30 seconds
        self.active_apis = {}
        self.keep_warm = True
        self.heartbeat_interval = 30  # seconds

    def start_and_keep_warm(self):
        """
        A method to:
          1) Run an async check/deploy of all apis_needed (blocking until done).
          2) Launch a background thread that does keep-warm indefinitely.
        """
        # Step 1: Do deployment checks in a single run call (this blocks)
        asyncio.run(self._initialize_apis())

        # Step 2: Once that finishes, spawn a background thread that keeps the apis needed warm in an *event loop* forever.
        t = Thread(target=self._keep_warm_loop_runner, daemon=True)
        t.start()

    async def _initialize_apis(self):
        """
        Async method that runs:
          - concurrency check of all apis needed
          - if needed, deploy them
        """
        print("Starting APIs initialization process...\n")

        tasks = [self._check_and_deploy(api_key, config) for api_key, config in self.apis_needed.items()]
        results = await asyncio.gather(*tasks)

        for api_key, instance in results:
            self.active_apis[api_key] = instance

        print("All APIs are deployed and ready!!")

    async def _check_and_deploy(self, api_key: str, config: dict):
        """
        Checks if the API is deployed by pinging it.
        If not found, deploys the API.
        Returns a tuple (api_key, instance).
        """
        try:
            print(f"Checking if {config['class_name']} API is deployed.")
            cls = modal.Cls.from_name(config['app_name'], config['class_name'])
            # Ping in a thread (so the event loop is not blocked)
            await asyncio.to_thread(lambda: cls().ping.remote())

        except NotFoundError:
            print(f"    {config['class_name']} API not found. Deploying...")
            # Attempt to deploy the API
            if not await self._deploy_api(api_key):
                raise RuntimeError(f"Deployment failed for {api_key}.")

            # Once deployed, re-instantiate the class
            cls = modal.Cls.from_name(config['app_name'], config['class_name'])

        return api_key, cls()

    async def _deploy_api(self, api_key: str) -> bool:
        """Deploy a specific API using the Modal CLI, asynchronously."""
        result = await asyncio.to_thread(
            subprocess.run,
            ['modal', 'deploy', f"apis.py::{self.apis_needed[api_key]['app_var']}"],
            capture_output=True,
            text=True
        )
        if result.returncode != 0:
            print(f"Failed to deploy {api_key}: {result.stderr}")
            return False

        print(f"    {self.apis_needed[api_key]['class_name']} API deployed successfully.")
        return True

    def _keep_warm_loop_runner(self):
        """
        **Runs in a background thread** and creates an asyncio event loop
        for the keep-warm coroutine that will run forever (or until self.keep_warm=False).
        """
        asyncio.run(self._keep_warm_loop())

    async def _keep_warm_loop(self):
        """Async keep-warm that runs forever, pinging each API every 30s."""
        print("Keep-warm process has started.")
        while self.keep_warm:
            tasks = [
                asyncio.to_thread(instance.ping.remote)
                for instance in self.active_apis.values()
            ]
            # Wait for all pings to complete
            await asyncio.gather(*tasks)
            await asyncio.sleep(self.heartbeat_interval)
