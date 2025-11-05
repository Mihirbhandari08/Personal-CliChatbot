#!/usr/bin/env python3
"""
DevBot - Your CLI Software Engineering Assistant
Supports multi-model consensus with async queries to multiple providers
"""

import os
import sys
import json
import argparse
import asyncio
from pathlib import Path
from datetime import datetime
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt, Confirm
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich import box
from rich.theme import Theme
from rich.columns import Columns
from rich.live import Live

# Import API clients
try:
    from groq import Groq
except ImportError:
    Groq = None

try:
    import google.generativeai as genai
except ImportError:
    genai = None

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

try:
    import requests
except ImportError:
    requests = None

custom_theme = Theme({
    "info": "cyan",
    "warning": "yellow",
    "error": "bold red",
    "success": "bold green",
    "question": "bold magenta",
    "answer": "bold cyan"
})

console = Console(theme=custom_theme)

class DevBot:
    def __init__(self, skip_api_check=False):
        self.config_dir = Path.home() / ".devbot"
        self.config_file = self.config_dir / "config.json"
        self.history_file = self.config_dir / "history.json"
        self.snippets_file = self.config_dir / "snippets.json"
        self.api_collections_file = self.config_dir / "api_collections.json"
        self.learning_progress_file = self.config_dir / "learning_progress.json"
        self.config_dir.mkdir(exist_ok=True)
        
        # File context for working with local files
        self.current_files = {}  # filepath: content
        self.file_context = []  # For AI context
        
        self.config = self.load_config()
        self.history = self.load_history()
        self.snippets = self.load_snippets()
        self.api_collections = self.load_api_collections()
        self.learning_progress = self.load_learning_progress()
        
        # Initialize clients dictionary BEFORE using it
        self.clients = {}
        self.chat_history = []
        
        # Set default settings
        if "provider" not in self.config:
            self.config["provider"] = "google"  # Changed to google as default
        if "multi_model_mode" not in self.config:
            self.config["multi_model_mode"] = False
        if "consensus_providers" not in self.config:
            self.config["consensus_providers"] = ["google", "groq"]
        
        self.current_provider = self.config.get("provider", "google")
        self.multi_model_mode = self.config.get("multi_model_mode", False)
        
        # Only check API key if we're not configuring it
        if not skip_api_check:
            self.check_and_setup_provider()
    
    def load_config(self):
        if self.config_file.exists():
            with open(self.config_file, 'r') as f:
                return json.load(f)
        return {"provider": "google", "api_keys": {}, "multi_model_mode": False}
    
    def save_config(self):
        with open(self.config_file, 'w') as f:
            json.dump(self.config, f, indent=2)
    
    def load_history(self):
        if self.history_file.exists():
            with open(self.history_file, 'r') as f:
                return json.load(f)
        return []
    
    def save_history(self):
        with open(self.history_file, 'w') as f:
            json.dump(self.history[-50:], f, indent=2)
    
    def load_snippets(self):
        if self.snippets_file.exists():
            with open(self.snippets_file, 'r') as f:
                return json.load(f)
        return {}
    
    def save_snippets(self):
        with open(self.snippets_file, 'w') as f:
            json.dump(self.snippets, f, indent=2)
    
    def load_api_collections(self):
        if self.api_collections_file.exists():
            with open(self.api_collections_file, 'r') as f:
                return json.load(f)
        return {}
    
    def save_api_collections(self):
        with open(self.api_collections_file, 'w') as f:
            json.dump(self.api_collections, f, indent=2)
    
    def load_learning_progress(self):
        if self.learning_progress_file.exists():
            with open(self.learning_progress_file, 'r') as f:
                return json.load(f)
        return {"topics": {}, "quizzes": {}, "total_score": 0}
    
    def save_learning_progress(self):
        with open(self.learning_progress_file, 'w') as f:
            json.dump(self.learning_progress, f, indent=2)
    
    def check_and_setup_provider(self):
        """Check if API key exists for current provider"""
        api_keys = self.config.get("api_keys", {})
        
        if self.multi_model_mode:
            # Check if all consensus providers have API keys
            consensus_providers = self.config.get("consensus_providers", [])
            missing = [p for p in consensus_providers if p not in api_keys or not api_keys[p]]
            
            if missing:
                console.print(Panel.fit(
                    f"[warning]⚠️  Multi-model mode requires API keys for: {', '.join(missing).upper()}[/warning]\n\n"
                    f"Set them up with:\n"
                    f"  [info]devbot config --provider <name> --set-key YOUR_KEY[/info]\n\n"
                    "[bold cyan]🆓 FREE API Key Sources:[/bold cyan]\n"
                    "• Google AI Studio: https://makersuite.google.com/app/apikey (1M tokens/day)\n"
                    "• Groq: https://console.groq.com (FASTEST, 14,400 req/day)",
                    title="🔑 API Keys Required",
                    border_style="yellow"
                ))
                sys.exit(1)
            
            # Initialize all consensus providers
            for provider in consensus_providers:
                self.initialize_provider(provider)
        else:
            if self.current_provider not in api_keys or not api_keys[self.current_provider]:
                console.print(Panel.fit(
                    f"[warning]⚠️  No API key found for {self.current_provider.upper()}![/warning]\n\n"
                    f"Set it up with:\n"
                    f"  [info]devbot config --set-key YOUR_API_KEY[/info]\n\n"
                    f"Or switch provider:\n"
                    f"  [info]devbot config --provider <google|groq|together|openrouter>[/info]\n\n"
                    "[bold cyan]🆓 FREE API Key Sources:[/bold cyan]\n"
                    "• Google AI Studio: https://makersuite.google.com/app/apikey (FREE - 1M tokens/day)\n"
                    "• Groq: https://console.groq.com (FASTEST - 14,400 req/day)\n"
                    "• Together AI: https://api.together.xyz (Free $25 credits)\n"
                    "• OpenRouter: https://openrouter.ai (Multiple models, some free)",
                    title="🔑 API Key Required",
                    border_style="yellow"
                ))
                sys.exit(1)
            
            self.initialize_provider(self.current_provider)
    
    def initialize_provider(self, provider_name):
        """Initialize a specific AI provider"""
        api_key = self.config["api_keys"].get(provider_name)
        
        if provider_name == "groq":
            if Groq is None:
                console.print("[error]Install groq: pip install groq[/error]")
                return False
            self.clients["groq"] = {
                "client": Groq(api_key=api_key),
                "model": "llama-3.3-70b-versatile",
                "type": "groq"
            }
            
        elif provider_name == "google":
            if genai is None:
                console.print("[error]Install google-generativeai: pip install google-generativeai[/error]")
                return False
            genai.configure(api_key=api_key)
            self.clients["google"] = {
              "client": genai.GenerativeModel('gemini-2.5-flash'),
                "model": "gemini-2.5-flash",
                "type": "google"
            }
            
        elif provider_name == "together":
            if OpenAI is None:
                console.print("[error]Install openai: pip install openai[/error]")
                return False
            self.clients["together"] = {
                "client": OpenAI(api_key=api_key, base_url="https://api.together.xyz/v1"),
                "model": "meta-llama/Llama-3.3-70B-Instruct-Turbo",
                "type": "openai"
            }
            
        elif provider_name == "openrouter":
            if OpenAI is None:
                console.print("[error]Install openai: pip install openai[/error]")
                return False
            self.clients["openrouter"] = {
                "client": OpenAI(api_key=api_key, base_url="https://openrouter.ai/api/v1"),
                "model": "meta-llama/llama-3.3-70b-instruct:free",
                "type": "openai"
            }
        
        return True
    
    async def query_provider_async(self, provider_name, question, system_prompt):
        """Query a single provider asynchronously"""
        try:
            provider = self.clients.get(provider_name)
            if not provider:
                return None, f"Provider {provider_name} not initialized"
            
            client = provider["client"]
            model = provider["model"]
            provider_type = provider["type"]
            
            if provider_type == "groq" or provider_type == "openai":
                messages = [
                    {"role": "system", "content": system_prompt},
                    *self.chat_history,
                    {"role": "user", "content": question}
                ]
                
                # Run in thread pool for blocking I/O
                loop = asyncio.get_event_loop()
                response = await loop.run_in_executor(
                    None,
                    lambda: client.chat.completions.create(
                        model=model,
                        messages=messages,
                        temperature=0.7,
                        max_tokens=2048
                    )
                )
                return response.choices[0].message.content, None
                
            elif provider_type == "google":
                full_prompt = f"{system_prompt}\n\nUser: {question}"
                loop = asyncio.get_event_loop()
                response = await loop.run_in_executor(
                    None,
                    lambda: client.generate_content(full_prompt)
                )
                return response.text, None
                
        except Exception as e:
            console.print(f"[dim red]Error from {provider_name}: {str(e)}[/dim red]")
            return None, str(e)
    
    def synthesize_responses(self, responses):
        """Combine multiple responses into a single best answer"""
        valid_responses = [(p, r) for p, r, e in responses if r is not None]
        
        if not valid_responses:
            # Show errors if all failed
            console.print("\n[bold red]All providers failed:[/bold red]")
            for provider, response, error in responses:
                if error:
                    console.print(f"  • {provider.upper()}: {error}")
            return "❌ All providers failed to respond. Please check your API keys and internet connection.", []
        
        if len(valid_responses) == 1:
            return valid_responses[0][1], [valid_responses[0][0]]
        
        # Use first provider to create synthesis prompt
        provider_outputs = "\n\n".join([
            f"=== {provider.upper()} Response ===\n{response}"
            for provider, response in valid_responses
        ])
        
        synthesis_prompt = f"""You are analyzing multiple AI responses to synthesize the best answer.

Question: {self.last_question}

{provider_outputs}

Your task:
1. Identify the most accurate, complete, and helpful information from all responses
2. Synthesize them into ONE comprehensive answer that takes the best from each
3. If responses conflict, use your judgment to determine the most correct approach
4. Keep the tone professional and technical
5. Include code examples if present in any response

Provide ONLY the synthesized answer, no meta-commentary about the synthesis process."""

        try:
            # PRIORITY: Use the best model for synthesis
            # 1st choice: Google Gemini (best reasoning and comprehension)
            # 2nd choice: Groq (fast and reliable)
            # 3rd choice: Any available provider
            
            synthesis_provider = None
            if "google" in self.clients:
                synthesis_provider = "google"
                console.print("[dim cyan]Using Gemini for synthesis (best reasoning)[/dim cyan]")
            elif "groq" in self.clients:
                synthesis_provider = "groq"
                console.print("[dim yellow]Using Groq for synthesis[/dim yellow]")
            else:
                # Fallback to first available provider
                synthesis_provider = valid_responses[0][0]
                console.print(f"[dim]Using {synthesis_provider.upper()} for synthesis[/dim]")
            
            provider = self.clients[synthesis_provider]
            
            if provider["type"] == "groq" or provider["type"] == "openai":
                response = provider["client"].chat.completions.create(
                    model=provider["model"],
                    messages=[{"role": "user", "content": synthesis_prompt}],
                    temperature=0.5,
                    max_tokens=3000
                )
                synthesized = response.choices[0].message.content
            elif provider["type"] == "google":
                response = provider["client"].generate_content(synthesis_prompt)
                synthesized = response.text
            
            used_providers = [p for p, _ in valid_responses]
            return synthesized, used_providers
            
        except Exception as e:
            console.print(f"[yellow]Synthesis failed: {e}[/yellow]")
            # Fallback: return the longest response
            longest = max(valid_responses, key=lambda x: len(x[1]))
            return longest[1], [longest[0]]
    
    async def ask_multi_model(self, question):
        """Query multiple models and synthesize the best response"""
        system_prompt = """You are DevBot, an expert software engineering assistant. You help developers with:
- Code explanations and debugging
- Algorithm design and optimization
- Best practices and design patterns
- Framework and library usage
- System design and architecture
- Code review and suggestions

Provide clear, concise, and practical answers. Include code examples when helpful.
Format code blocks with proper syntax highlighting using markdown."""
        
        self.last_question = question
        consensus_providers = self.config.get("consensus_providers", ["google", "groq"])
        
        # Query all providers in parallel
        tasks = [
            self.query_provider_async(provider, question, system_prompt)
            for provider in consensus_providers
            if provider in self.clients
        ]
        
        # Wait for all responses
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Format results
        responses = []
        for i, result in enumerate(results):
            provider = consensus_providers[i] if i < len(consensus_providers) else "unknown"
            if isinstance(result, Exception):
                responses.append((provider, None, str(result)))
            else:
                answer, error = result
                responses.append((provider, answer, error))
        
        return responses
    
    def ask(self, question, save_to_history=True):
        """Ask a question and get a response"""
        # Build system prompt with file context if available
        file_context_str = ""
        if self.file_context:
            file_context_str = "\n\nCurrent files in context:\n"
            for filepath, content in self.file_context:
                file_context_str += f"\n--- File: {filepath} ---\n{content}\n"
        
        system_prompt = f"""You are DevBot, an expert software engineering assistant. You help developers with:
- Code explanations and debugging
- Algorithm design and optimization
- Best practices and design patterns
- Framework and library usage
- System design and architecture
- Code review and suggestions
- File operations and modifications

Provide clear, concise, and practical answers. Include code examples when helpful.
Format code blocks with proper syntax highlighting using markdown.{file_context_str}"""
        
        try:
            if self.multi_model_mode:
                # Multi-model consensus mode
                with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    console=console,
                    transient=True
                ) as progress:
                    providers = self.config.get("consensus_providers", [])
                    progress.add_task(
                        f"🧠 Querying {len(providers)} models simultaneously...", 
                        total=None
                    )
                    
                    # Run async queries
                    responses = asyncio.run(self.ask_multi_model(question))
                
                # Show individual responses (optional - can be toggled)
                if self.config.get("show_individual_responses", False):
                    console.print("\n[dim]━━━ Individual Model Responses ━━━[/dim]\n")
                    for provider, response, error in responses:
                        if response:
                            console.print(Panel(
                                Markdown(response[:300] + "..." if len(response) > 300 else response),
                                title=f"[cyan]{provider.upper()}[/cyan]",
                                border_style="dim cyan",
                                padding=(0, 1)
                            ))
                
                # Synthesize best answer
                console.print("\n[bold cyan]🔄 Synthesizing best response...[/bold cyan]\n")
                answer, used_providers = self.synthesize_responses(responses)
                
                # Update metadata
                metadata = {
                    "mode": "multi_model",
                    "providers": used_providers,
                    "individual_responses": len([r for _, r, _ in responses if r])
                }
            else:
                # Single model mode
                with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    console=console,
                    transient=True
                ) as progress:
                    progress.add_task(f"🧠 Thinking ({self.current_provider.upper()})...", total=None)
                    
                    provider = self.clients[self.current_provider]
                    
                    if provider["type"] == "groq" or provider["type"] == "openai":
                        messages = [
                            {"role": "system", "content": system_prompt},
                            *self.chat_history,
                            {"role": "user", "content": question}
                        ]
                        response = provider["client"].chat.completions.create(
                            model=provider["model"],
                            messages=messages,
                            temperature=0.7,
                            max_tokens=2048
                        )
                        answer = response.choices[0].message.content
                        
                    elif provider["type"] == "google":
                        full_prompt = f"{system_prompt}\n\nUser: {question}"
                        response = provider["client"].generate_content(full_prompt)
                        answer = response.text
                
                metadata = {"mode": "single", "provider": self.current_provider}
            
            # Update chat history
            self.chat_history.append({"role": "user", "content": question})
            self.chat_history.append({"role": "assistant", "content": answer})
            
            # Keep only last 20 messages
            if len(self.chat_history) > 20:
                self.chat_history = self.chat_history[-20:]
            
            if save_to_history:
                self.history.append({
                    "timestamp": datetime.now().isoformat(),
                    "question": question,
                    "answer": answer,
                    **metadata
                })
                self.save_history()
            
            return answer, metadata
            
        except Exception as e:
            return f"❌ Error: {str(e)}", {"mode": "error"}
    
    def set_api_key(self, key, provider=None):
        """Set API key for a specific provider"""
        target_provider = provider if provider else self.current_provider
        if "api_keys" not in self.config:
            self.config["api_keys"] = {}
        self.config["api_keys"][target_provider] = key
        self.save_config()
        console.print(f"[success]✅ API key saved for {target_provider.upper()}![/success]")
    
    def set_provider(self, provider):
        """Switch AI provider"""
        valid_providers = ["google", "groq", "together", "openrouter"]
        if provider not in valid_providers:
            console.print(f"[error]Invalid provider. Choose from: {', '.join(valid_providers)}[/error]")
            return
        
        self.config["provider"] = provider
        self.current_provider = provider
        self.save_config()
        console.print(f"[success]✅ Switched to {provider.upper()}![/success]")
    
    def toggle_multi_model(self, enable=None):
        """Toggle multi-model consensus mode"""
        if enable is None:
            self.config["multi_model_mode"] = not self.config.get("multi_model_mode", False)
        else:
            self.config["multi_model_mode"] = enable
        
        self.multi_model_mode = self.config["multi_model_mode"]
        self.save_config()
        
        status = "ENABLED" if self.multi_model_mode else "DISABLED"
        console.print(f"[success]✅ Multi-model mode {status}![/success]")
        
        if self.multi_model_mode:
            providers = self.config.get("consensus_providers", [])
            console.print(f"[info]Using providers: {', '.join([p.upper() for p in providers])}[/info]")
    
    def interactive_mode(self):
        """Start interactive chat mode"""
        mode_indicator = "🔀 MULTI-MODEL" if self.multi_model_mode else f"📡 {self.current_provider.upper()}"
        
        console.print(Panel.fit(
            "[bold cyan]DevBot[/bold cyan] - Your CLI Coding Assistant 🤖\n\n"
            f"[info]Mode: {mode_indicator}[/info]\n\n"
            "💡 [info]Ask me anything about software engineering and coding![/info]\n\n"
            "[dim]Commands:[/dim]\n"
            "  [bold]/exit[/bold]           - Exit DevBot\n"
            "  [bold]/clear[/bold]          - Clear chat history\n"
            "  [bold]/history[/bold]        - View recent conversations\n"
            "  [bold]/save[/bold] <file>    - Save conversation to file\n"
            "  [bold]/snippet[/bold] <n> - Save last answer as snippet\n"
            "  [bold]/snippets[/bold]       - List saved snippets\n"
            "  [bold]/stats[/bold]          - View usage statistics\n"
            "  [bold]/mode[/bold]           - Toggle multi-model mode\n"
            "  [bold]/provider[/bold]       - Show current provider\n"
            "  [bold]/api[/bold]            - API testing mode\n"
            "  [bold]/file[/bold]           - File operations mode\n"
            "  [bold]/learn[/bold]          - Learning mode\n"
            "  [bold]/help[/bold]           - Show this help message",
            title="🚀 Welcome to DevBot",
            border_style="cyan",
            box=box.DOUBLE
        ))
        
        conversation_log = []
        last_answer = None
        
        while True:
            try:
                console.print()
                question = Prompt.ask("[question]🧑‍💻 You[/question]").strip()
                
                if not question:
                    continue
                
                # Handle commands
                if question == "/exit":
                    if Confirm.ask("\n[warning]Are you sure you want to exit?[/warning]"):
                        console.print("\n[success]👋 Goodbye! Happy coding![/success]")
                        break
                    continue
                    
                elif question == "/clear":
                    self.chat_history = []
                    conversation_log = []
                    last_answer = None
                    console.print("[success]✅ Chat cleared. Starting fresh![/success]")
                    continue
                    
                elif question == "/history":
                    self.show_history()
                    continue
                    
                elif question.startswith("/save "):
                    filename = question.split(" ", 1)[1]
                    self.save_conversation(conversation_log, filename)
                    continue
                    
                elif question.startswith("/snippet "):
                    if last_answer:
                        name = question.split(" ", 1)[1]
                        self.save_snippet(name, last_answer)
                    else:
                        console.print("[error]No answer to save as snippet![/error]")
                    continue
                    
                elif question == "/snippets":
                    self.show_snippets()
                    continue
                    
                elif question == "/stats":
                    self.show_stats()
                    continue
                    
                elif question == "/mode":
                    self.toggle_multi_model()
                    continue
                    
                elif question == "/provider":
                    self.show_provider_info()
                    continue
                    
                elif question == "/api":
                    self.api_testing_mode()
                    continue
                    
                elif question == "/file":
                    self.file_operations_mode()
                    continue
                    
                elif question == "/learn":
                    self.learning_mode()
                    continue
                    
                elif question == "/help":
                    self.show_help()
                    continue
                
                # Get answer
                answer, metadata = self.ask(question)
                last_answer = answer
                
                # Display answer with rich formatting
                title_text = "[answer]🤖 DevBot"
                if metadata.get("mode") == "multi_model":
                    providers = metadata.get("providers", [])
                    title_text += f" (Synthesized from {', '.join([p.upper() for p in providers])})"
                elif metadata.get("mode") == "single":
                    title_text += f" ({metadata.get('provider', '').upper()})"
                title_text += "[/answer]"
                
                console.print(Panel(
                    Markdown(answer),
                    title=title_text,
                    border_style="cyan",
                    padding=(1, 2)
                ))
                
                conversation_log.append({"Q": question, "A": answer})
                
            except KeyboardInterrupt:
                console.print("\n\n[success]👋 Goodbye! Happy coding![/success]")
                break
            except Exception as e:
                console.print(f"\n[error]❌ Error: {e}[/error]")
    
    def show_provider_info(self):
        """Show current provider information"""
        mode_text = "[bold green]ENABLED[/bold green]" if self.multi_model_mode else "[bold red]DISABLED[/bold red]"
        
        if self.multi_model_mode:
            providers = self.config.get("consensus_providers", [])
            providers_text = ", ".join([p.upper() for p in providers])
            
            info_text = f"""
[bold cyan]Multi-Model Mode:[/bold cyan] {mode_text}
[bold magenta]Active Providers:[/bold magenta] {providers_text}
[bold yellow]How it works:[/bold yellow]
  • Queries all models simultaneously (async)
  • Synthesizes the best response from all answers
  • Combines strengths of multiple AI models
  • More accurate and comprehensive answers
            """
        else:
            providers_info = {
                "google": "Gemini 1.5 Flash - Free, 1M tokens/day",
                "groq": "Llama 3.3 70B - Ultra Fast (300+ tok/s)",
                "together": "Llama 3.3 70B Turbo - Fast",
                "openrouter": "Llama 3.3 70B - Flexible"
            }
            
            info_text = f"""
[bold cyan]Multi-Model Mode:[/bold cyan] {mode_text}
[bold cyan]Current Provider:[/bold cyan] {self.current_provider.upper()}
[bold magenta]Model:[/bold magenta] {providers_info.get(self.current_provider, 'Unknown')}

[dim]Tip: Enable multi-model mode with /mode command[/dim]
            """
        
        console.print(Panel(
            info_text.strip(),
            title="🔌 Provider Information",
            border_style="blue",
            box=box.ROUNDED
        ))

    def api_testing_mode(self):
        """Interactive API testing mode"""
        if requests is None:
            console.print("[error]Please install requests: pip install requests[/error]")
            return
        
        console.print(Panel.fit(
            "[bold cyan]🌐 API Testing Mode[/bold cyan]\n\n"
            "Test REST APIs directly from CLI!\n\n"
            "[dim]Commands:[/dim]\n"
            "  [bold]get <url>[/bold]                    - GET request\n"
            "  [bold]post <url> <json>[/bold]            - POST request with JSON\n"
            "  [bold]put <url> <json>[/bold]             - PUT request with JSON\n"
            "  [bold]delete <url>[/bold]                 - DELETE request\n"
            "  [bold]headers <key:value>[/bold]          - Add custom header\n"
            "  [bold]clear-headers[/bold]                - Clear all headers\n"
            "  [bold]save <n> <url> <method>[/bold]   - Save API to collection\n"
            "  [bold]list[/bold]                         - List saved APIs\n"
            "  [bold]run <n>[/bold]                   - Run saved API\n"
            "  [bold]back[/bold]                         - Back to DevBot\n"
            "  [bold]exit[/bold]                         - Exit DevBot",
            title="🚀 API Testing",
            border_style="green",
            box=box.DOUBLE
        ))
        
        headers = {}
        
        while True:
            try:
                console.print()
                cmd = Prompt.ask("[bold green]API[/bold green]").strip()
                
                if not cmd:
                    continue
                
                parts = cmd.split(maxsplit=1)
                command = parts[0].lower()
                args = parts[1] if len(parts) > 1 else ""
                
                if command == "back":
                    console.print("[success]↩️  Returning to DevBot...[/success]")
                    break
                
                elif command == "exit":
                    if Confirm.ask("\n[warning]Are you sure you want to exit?[/warning]"):
                        console.print("\n[success]👋 Goodbye![/success]")
                        sys.exit(0)
                    continue
                
                elif command == "headers":
                    if ":" not in args:
                        console.print("[error]Format: headers Key:Value[/error]")
                        continue
                    key, value = args.split(":", 1)
                    headers[key.strip()] = value.strip()
                    console.print(f"[success]✅ Header added: {key.strip()}[/success]")
                
                elif command == "clear-headers":
                    headers = {}
                    console.print("[success]✅ Headers cleared[/success]")
                
                elif command == "get":
                    self.make_api_request("GET", args.strip(), headers=headers)
                
                elif command == "post":
                    parts = args.split(maxsplit=1)
                    if len(parts) < 2:
                        console.print("[error]Format: post <url> <json_data>[/error]")
                        continue
                    url, data = parts
                    self.make_api_request("POST", url.strip(), data=data, headers=headers)
                
                elif command == "put":
                    parts = args.split(maxsplit=1)
                    if len(parts) < 2:
                        console.print("[error]Format: put <url> <json_data>[/error]")
                        continue
                    url, data = parts
                    self.make_api_request("PUT", url.strip(), data=data, headers=headers)
                
                elif command == "delete":
                    self.make_api_request("DELETE", args.strip(), headers=headers)
                
                elif command == "save":
                    parts = args.split(maxsplit=2)
                    if len(parts) < 3:
                        console.print("[error]Format: save <name> <url> <method>[/error]")
                        continue
                    name, url, method = parts
                    self.save_api_to_collection(name, url, method.upper(), headers)
                
                elif command == "list":
                    self.list_api_collections()
                
                elif command == "run":
                    self.run_saved_api(args.strip())
                
                else:
                    console.print("[error]Unknown command. Type 'back' to return.[/error]")
                    
            except KeyboardInterrupt:
                console.print("\n[info]Use 'back' to return to DevBot[/info]")
            except Exception as e:
                console.print(f"[error]❌ Error: {e}[/error]")
    
    def make_api_request(self, method, url, data=None, headers=None):
        """Make an API request and display results"""
        if not url.startswith(("http://", "https://")):
            url = "https://" + url
        
        try:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
                transient=True
            ) as progress:
                progress.add_task(f"🌐 {method} {url}...", total=None)
                
                start_time = datetime.now()
                
                if method == "GET":
                    response = requests.get(url, headers=headers, timeout=10)
                elif method == "POST":
                    json_data = json.loads(data) if data else {}
                    response = requests.post(url, json=json_data, headers=headers, timeout=10)
                elif method == "PUT":
                    json_data = json.loads(data) if data else {}
                    response = requests.put(url, json=json_data, headers=headers, timeout=10)
                elif method == "DELETE":
                    response = requests.delete(url, headers=headers, timeout=10)
                
                elapsed = (datetime.now() - start_time).total_seconds()
            
            # Status code with color
            status_color = "green" if 200 <= response.status_code < 300 else "yellow" if 300 <= response.status_code < 400 else "red"
            
            # Create result table
            table = Table(title=f"📡 {method} Response", box=box.ROUNDED, show_header=False)
            table.add_column("Property", style="cyan", width=15)
            table.add_column("Value", style="white")
            
            table.add_row("Status", f"[{status_color}]{response.status_code} {response.reason}[/{status_color}]")
            table.add_row("Time", f"{elapsed:.2f}s")
            table.add_row("Size", f"{len(response.content)} bytes")
            
            console.print(table)
            
            # Headers
            if response.headers:
                console.print("\n[bold cyan]Response Headers:[/bold cyan]")
                headers_table = Table(box=box.SIMPLE, show_header=False, padding=(0, 1))
                headers_table.add_column("Key", style="dim cyan")
                headers_table.add_column("Value", style="white")
                
                for key, value in list(response.headers.items())[:10]:
                    headers_table.add_row(key, value[:80] + "..." if len(value) > 80 else value)
                
                console.print(headers_table)
            
            # Body
            console.print("\n[bold cyan]Response Body:[/bold cyan]")
            try:
                json_response = response.json()
                formatted_json = json.dumps(json_response, indent=2)
                
                # Syntax highlighting for JSON
                if len(formatted_json) > 2000:
                    console.print(Panel(
                        formatted_json[:2000] + f"\n\n[dim]... ({len(formatted_json) - 2000} more characters)[/dim]",
                        border_style="dim blue",
                        padding=(1, 2)
                    ))
                else:
                    console.print(Panel(
                        Markdown(f"```json\n{formatted_json}\n```"),
                        border_style="blue",
                        padding=(0, 1)
                    ))
            except:
                # Not JSON, display as text
                text = response.text
                if len(text) > 1000:
                    console.print(Panel(
                        text[:1000] + f"\n\n[dim]... ({len(text) - 1000} more characters)[/dim]",
                        border_style="dim blue",
                        padding=(1, 2)
                    ))
                else:
                    console.print(Panel(text, border_style="blue", padding=(1, 2)))
                    
        except requests.exceptions.Timeout:
            console.print("[error]❌ Request timed out (10s)[/error]")
        except requests.exceptions.ConnectionError:
            console.print("[error]❌ Connection error. Check URL and internet.[/error]")
        except json.JSONDecodeError:
            console.print("[error]❌ Invalid JSON in request body[/error]")
        except Exception as e:
            console.print(f"[error]❌ Error: {e}[/error]")
    
    def save_api_to_collection(self, name, url, method, headers=None):
        """Save API endpoint to collection"""
        self.api_collections[name] = {
            "url": url,
            "method": method,
            "headers": headers or {},
            "created": datetime.now().isoformat()
        }
        self.save_api_collections()
        console.print(f"[success]✅ API '{name}' saved to collection![/success]")
    
    def list_api_collections(self):
        """List all saved API endpoints"""
        if not self.api_collections:
            console.print("[warning]📝 No saved APIs yet.[/warning]")
            return
        
        table = Table(title="📚 Saved API Collections", 
                     box=box.ROUNDED,
                     show_header=True,
                     header_style="bold cyan")
        table.add_column("Name", style="magenta", width=20)
        table.add_column("Method", style="green", width=8)
        table.add_column("URL", style="cyan")
        
        for name, data in self.api_collections.items():
            url = data["url"][:50] + "..." if len(data["url"]) > 50 else data["url"]
            table.add_row(name, data["method"], url)
        
        console.print(table)
    
    def run_saved_api(self, name):
        """Run a saved API endpoint"""
        if name not in self.api_collections:
            console.print(f"[error]API '{name}' not found. Use 'list' to see saved APIs.[/error]")
            return
        
        api = self.api_collections[name]
        console.print(f"[info]Running: {api['method']} {api['url']}[/info]")
        self.make_api_request(api["method"], api["url"], headers=api.get("headers"))
    
    def file_operations_mode(self):
        """Interactive file operations mode"""
        console.print(Panel.fit(
            "[bold cyan]📁 File Operations Mode[/bold cyan]\n\n"
            "Work with local files and get AI assistance!\n\n"
            "[dim]Commands:[/dim]\n"
            "  [bold]load <path>[/bold]                  - Load file into context\n"
            "  [bold]read <path>[/bold]                  - Read and display file\n"
            "  [bold]write <path> <content>[/bold]       - Write content to file\n"
            "  [bold]append <path> <content>[/bold]      - Append to file\n"
            "  [bold]create <path>[/bold]                - Create new file\n"
            "  [bold]delete <path>[/bold]                - Delete file\n"
            "  [bold]list[/bold]                         - List loaded files\n"
            "  [bold]clear[/bold]                        - Clear file context\n"
            "  [bold]ask <question>[/bold]               - Ask AI about loaded files\n"
            "  [bold]analyze <path>[/bold]               - Full code analysis\n"
            "  [bold]fix <path> <issue>[/bold]           - AI fixes code issue\n"
            "  [bold]refactor <path>[/bold]              - AI refactors code\n"
            "  [bold]back[/bold]                         - Back to DevBot\n"
            "  [bold]exit[/bold]                         - Exit DevBot",
            title="🚀 File Operations",
            border_style="magenta",
            box=box.DOUBLE
        ))
        
        while True:
            try:
                console.print()
                
                # Show loaded files indicator
                if self.current_files:
                    files_indicator = f"[dim]({len(self.current_files)} files loaded)[/dim]"
                else:
                    files_indicator = ""
                
                cmd = Prompt.ask(f"[bold magenta]FILE[/bold magenta] {files_indicator}").strip()
                
                if not cmd:
                    continue
                
                parts = cmd.split(maxsplit=1)
                command = parts[0].lower()
                args = parts[1] if len(parts) > 1 else ""
                
                if command == "back":
                    console.print("[success]↩️  Returning to DevBot...[/success]")
                    break
                
                elif command == "exit":
                    if Confirm.ask("\n[warning]Are you sure you want to exit?[/warning]"):
                        console.print("\n[success]👋 Goodbye![/success]")
                        sys.exit(0)
                    continue
                
                elif command == "load":
                    self.load_file(args.strip())
                
                elif command == "read":
                    self.read_file(args.strip())
                
                elif command == "write":
                    parts = args.split(maxsplit=1)
                    if len(parts) < 2:
                        console.print("[error]Format: write <path> <content>[/error]")
                        continue
                    path, content = parts
                    self.write_file(path.strip(), content)
                
                elif command == "append":
                    parts = args.split(maxsplit=1)
                    if len(parts) < 2:
                        console.print("[error]Format: append <path> <content>[/error]")
                        continue
                    path, content = parts
                    self.append_file(path.strip(), content)
                
                elif command == "create":
                    self.create_file(args.strip())
                
                elif command == "delete":
                    self.delete_file(args.strip())
                
                elif command == "list":
                    self.list_loaded_files()
                
                elif command == "clear":
                    self.current_files = {}
                    self.file_context = []
                    console.print("[success]✅ File context cleared[/success]")
                
                elif command == "ask":
                    if not self.current_files:
                        console.print("[warning]Load files first with 'load <path>'[/warning]")
                        continue
                    answer, metadata = self.ask(args, save_to_history=False)
                    console.print(Panel(
                        Markdown(answer),
                        title="[cyan]🤖 DevBot[/cyan]",
                        border_style="cyan",
                        padding=(1, 2)
                    ))
                
                elif command == "analyze":
                    self.analyze_file(args.strip())
                
                elif command == "fix":
                    parts = args.split(maxsplit=1)
                    if len(parts) < 2:
                        console.print("[error]Format: fix <path> <issue_description>[/error]")
                        continue
                    path, issue = parts
                    self.fix_file(path.strip(), issue)
                
                elif command == "refactor":
                    self.refactor_file(args.strip())
                
                else:
                    console.print("[error]Unknown command. Type 'back' to return.[/error]")
                    
            except KeyboardInterrupt:
                console.print("\n[info]Use 'back' to return to DevBot[/info]")
            except Exception as e:
                console.print(f"[error]❌ Error: {e}[/error]")
    
    def load_file(self, filepath):
        """Load a file into context"""
        try:
            path = Path(filepath).expanduser()
            if not path.exists():
                console.print(f"[error]File not found: {filepath}[/error]")
                return
            
            if not path.is_file():
                console.print(f"[error]Not a file: {filepath}[/error]")
                return
            
            # Read file
            with open(path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Store in context
            self.current_files[str(path)] = content
            self.file_context.append((str(path), content))
            
            # Show file info
            lines = len(content.split('\n'))
            size = len(content)
            
            console.print(f"[success]✅ Loaded: {path.name}[/success]")
            console.print(f"[info]   Lines: {lines} | Size: {size} bytes[/info]")
            
        except UnicodeDecodeError:
            console.print("[error]Cannot read binary file. Please use text files only.[/error]")
        except Exception as e:
            console.print(f"[error]❌ Error loading file: {e}[/error]")
    
    def read_file(self, filepath):
        """Read and display file content"""
        try:
            path = Path(filepath).expanduser()
            if not path.exists():
                console.print(f"[error]File not found: {filepath}[/error]")
                return
            
            with open(path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Detect language for syntax highlighting
            ext = path.suffix.lower()
            lang_map = {
                '.py': 'python', '.js': 'javascript', '.ts': 'typescript',
                '.java': 'java', '.cpp': 'cpp', '.c': 'c', '.cs': 'csharp',
                '.rb': 'ruby', '.go': 'go', '.rs': 'rust', '.php': 'php',
                '.html': 'html', '.css': 'css', '.json': 'json', '.xml': 'xml',
                '.md': 'markdown', '.sql': 'sql', '.sh': 'bash', '.yml': 'yaml',
                '.yaml': 'yaml', '.toml': 'toml'
            }
            language = lang_map.get(ext, 'text')
            
            console.print(Panel(
                Markdown(f"```{language}\n{content}\n```"),
                title=f"[cyan]📄 {path.name}[/cyan]",
                border_style="cyan",
                padding=(0, 1)
            ))
            
        except Exception as e:
            console.print(f"[error]❌ Error reading file: {e}[/error]")
    
    def write_file(self, filepath, content):
        """Write content to file"""
        try:
            path = Path(filepath).expanduser()
            
            # Confirm overwrite if exists
            if path.exists():
                if not Confirm.ask(f"[warning]File exists. Overwrite {path.name}?[/warning]"):
                    console.print("[info]Operation cancelled[/info]")
                    return
            
            # Create parent directories if needed
            path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            console.print(f"[success]✅ Written to: {path}[/success]")
            
            # Update context if file was loaded
            if str(path) in self.current_files:
                self.current_files[str(path)] = content
                self.file_context = [(p, c) for p, c in self.file_context if p != str(path)]
                self.file_context.append((str(path), content))
            
        except Exception as e:
            console.print(f"[error]❌ Error writing file: {e}[/error]")
    
    def append_file(self, filepath, content):
        """Append content to file"""
        try:
            path = Path(filepath).expanduser()
            
            if not path.exists():
                console.print(f"[error]File not found: {filepath}[/error]")
                return
            
            with open(path, 'a', encoding='utf-8') as f:
                f.write('\n' + content)
            
            console.print(f"[success]✅ Appended to: {path}[/success]")
            
            # Update context
            if str(path) in self.current_files:
                with open(path, 'r', encoding='utf-8') as f:
                    new_content = f.read()
                self.current_files[str(path)] = new_content
                self.file_context = [(p, c) for p, c in self.file_context if p != str(path)]
                self.file_context.append((str(path), new_content))
            
        except Exception as e:
            console.print(f"[error]❌ Error appending to file: {e}[/error]")
    
    def create_file(self, filepath):
        """Create a new empty file"""
        try:
            path = Path(filepath).expanduser()
            
            if path.exists():
                console.print(f"[error]File already exists: {filepath}[/error]")
                return
            
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch()
            
            console.print(f"[success]✅ Created: {path}[/success]")
            
        except Exception as e:
            console.print(f"[error]❌ Error creating file: {e}[/error]")
    
    def delete_file(self, filepath):
        """Delete a file"""
        try:
            path = Path(filepath).expanduser()
            
            if not path.exists():
                console.print(f"[error]File not found: {filepath}[/error]")
                return
            
            if not Confirm.ask(f"[warning]⚠️  Delete {path.name}? This cannot be undone![/warning]"):
                console.print("[info]Operation cancelled[/info]")
                return
            
            path.unlink()
            console.print(f"[success]✅ Deleted: {path}[/success]")
            
            # Remove from context
            if str(path) in self.current_files:
                del self.current_files[str(path)]
                self.file_context = [(p, c) for p, c in self.file_context if p != str(path)]
            
        except Exception as e:
            console.print(f"[error]❌ Error deleting file: {e}[/error]")
    
    def list_loaded_files(self):
        """List all files currently in context"""
        if not self.current_files:
            console.print("[warning]📝 No files loaded yet.[/warning]")
            return
        
        table = Table(title="📚 Loaded Files", 
                     box=box.ROUNDED,
                     show_header=True,
                     header_style="bold cyan")
        table.add_column("#", style="dim", width=4)
        table.add_column("File", style="cyan")
        table.add_column("Lines", style="green", width=10)
        table.add_column("Size", style="yellow", width=12)
        
        for i, (filepath, content) in enumerate(self.current_files.items(), 1):
            filename = Path(filepath).name
            lines = len(content.split('\n'))
            size = len(content)
            size_str = f"{size} bytes" if size < 1024 else f"{size/1024:.1f} KB"
            table.add_row(str(i), filename, str(lines), size_str)
        
        console.print(table)
    
    def analyze_file(self, filepath):
        """AI-powered file analysis"""
        try:
            path = Path(filepath).expanduser()
            if not path.exists():
                console.print(f"[error]File not found: {filepath}[/error]")
                return
            
            # Load file if not already loaded
            if str(path) not in self.current_files:
                self.load_file(str(path))
            
            question = f"""Analyze the file '{path.name}' and provide:
1. Purpose and functionality overview
2. Code quality assessment
3. Potential bugs or issues
4. Performance considerations
5. Best practices violations
6. Suggestions for improvement
7. Security concerns (if any)

Be thorough but concise."""
            
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
                transient=True
            ) as progress:
                progress.add_task("🔍 Analyzing file...", total=None)
                answer, metadata = self.ask(question, save_to_history=False)
            
            console.print(Panel(
                Markdown(answer),
                title=f"[cyan]🔍 Analysis: {path.name}[/cyan]",
                border_style="cyan",
                padding=(1, 2)
            ))
            
        except Exception as e:
            console.print(f"[error]❌ Error analyzing file: {e}[/error]")
    
    def fix_file(self, filepath, issue):
        """AI fixes code issues"""
        try:
            path = Path(filepath).expanduser()
            if not path.exists():
                console.print(f"[error]File not found: {filepath}[/error]")
                return
            
            # Load file if not already loaded
            if str(path) not in self.current_files:
                self.load_file(str(path))
            
            question = f"""Fix the following issue in '{path.name}':

Issue: {issue}

Provide:
1. Explanation of the problem
2. Complete fixed code
3. Explanation of the fix

Format the fixed code in a code block so it can be easily copied."""
            
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
                transient=True
            ) as progress:
                progress.add_task("🔧 Fixing issue...", total=None)
                answer, metadata = self.ask(question, save_to_history=False)
            
            console.print(Panel(
                Markdown(answer),
                title=f"[green]🔧 Fix: {path.name}[/green]",
                border_style="green",
                padding=(1, 2)
            ))
            
            if Confirm.ask("\n[question]Apply this fix to the file?[/question]"):
                # Extract code from markdown code blocks
                import re
                code_blocks = re.findall(r'```(?:\w+)?\n(.*?)```', answer, re.DOTALL)
                if code_blocks:
                    self.write_file(str(path), code_blocks[0].strip())
                else:
                    console.print("[warning]No code block found in response. Please copy manually.[/warning]")
            
        except Exception as e:
            console.print(f"[error]❌ Error fixing file: {e}[/error]")
    
    def refactor_file(self, filepath):
        """AI refactors code"""
        try:
            path = Path(filepath).expanduser()
            if not path.exists():
                console.print(f"[error]File not found: {filepath}[/error]")
                return
            
            # Load file if not already loaded
            if str(path) not in self.current_files:
                self.load_file(str(path))
            
            question = f"""Refactor the code in '{path.name}' to improve:
1. Code readability and clarity
2. Performance and efficiency
3. Best practices and design patterns
4. Error handling
5. Documentation and comments

Provide the complete refactored code with explanations of key changes."""
            
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
                transient=True
            ) as progress:
                progress.add_task("♻️ Refactoring...", total=None)
                answer, metadata = self.ask(question, save_to_history=False)
            
            console.print(Panel(
                Markdown(answer),
                title=f"[yellow]♻️ Refactored: {path.name}[/yellow]",
                border_style="yellow",
                padding=(1, 2)
            ))
            
            if Confirm.ask("\n[question]Apply refactored code to the file?[/question]"):
                # Extract code from markdown code blocks
                import re
                code_blocks = re.findall(r'```(?:\w+)?\n(.*?)```', answer, re.DOTALL)
                if code_blocks:
                    self.write_file(str(path), code_blocks[0].strip())
                else:
                    console.print("[warning]No code block found in response. Please copy manually.[/warning]")
            
        except Exception as e:
            console.print(f"[error]❌ Error refactoring file: {e}[/error]")
    
    def learning_mode(self):
        """Interactive learning mode with tutorials, quizzes, and roadmaps"""
        console.print(Panel.fit(
            "[bold cyan]🎓 Learning Mode[/bold cyan]\n\n"
            "Interactive learning, tutorials, and skill development!\n\n"
            "[dim]Commands:[/dim]\n"
            "  [bold]topic <subject>[/bold]              - Learn about a topic\n"
            "  [bold]explain <concept>[/bold]            - Get detailed explanation\n"
            "  [bold]eli5 <concept>[/bold]               - Explain Like I'm 5\n"
            "  [bold]quiz <topic>[/bold]                 - Take a quiz on topic\n"
            "  [bold]practice <concept>[/bold]           - Get coding exercises\n"
            "  [bold]roadmap <career>[/bold]             - Get learning roadmap\n"
            "  [bold]progress[/bold]                     - View learning progress\n"
            "  [bold]topics[/bold]                       - List popular topics\n"
            "  [bold]back[/bold]                         - Back to DevBot\n"
            "  [bold]exit[/bold]                         - Exit DevBot",
            title="🚀 Learning Mode",
            border_style="blue",
            box=box.DOUBLE
        ))
        
        while True:
            try:
                console.print()
                cmd = Prompt.ask("[bold blue]LEARN[/bold blue]").strip()
                
                if not cmd:
                    continue
                
                parts = cmd.split(maxsplit=1)
                command = parts[0].lower()
                args = parts[1] if len(parts) > 1 else ""
                
                if command == "back":
                    console.print("[success]↩️  Returning to DevBot...[/success]")
                    break
                
                elif command == "exit":
                    if Confirm.ask("\n[warning]Are you sure you want to exit?[/warning]"):
                        console.print("\n[success]👋 Goodbye![/success]")
                        sys.exit(0)
                    continue
                
                elif command == "topic":
                    if not args:
                        console.print("[error]Usage: topic <subject>[/error]")
                        continue
                    self.learn_topic(args.strip())
                
                elif command == "explain":
                    if not args:
                        console.print("[error]Usage: explain <concept>[/error]")
                        continue
                    self.explain_concept(args.strip())
                
                elif command == "eli5":
                    if not args:
                        console.print("[error]Usage: eli5 <concept>[/error]")
                        continue
                    self.explain_like_five(args.strip())
                
                elif command == "quiz":
                    if not args:
                        console.print("[error]Usage: quiz <topic>[/error]")
                        continue
                    self.take_quiz(args.strip())
                
                elif command == "practice":
                    if not args:
                        console.print("[error]Usage: practice <concept>[/error]")
                        continue
                    self.practice_coding(args.strip())
                
                elif command == "roadmap":
                    if not args:
                        console.print("[error]Usage: roadmap <career/technology>[/error]")
                        continue
                    self.show_roadmap(args.strip())
                
                elif command == "progress":
                    self.show_learning_progress()
                
                elif command == "topics":
                    self.show_popular_topics()
                
                else:
                    console.print("[error]Unknown command. Type 'back' to return.[/error]")
                    
            except KeyboardInterrupt:
                console.print("\n[info]Use 'back' to return to DevBot[/info]")
            except Exception as e:
                console.print(f"[error]❌ Error: {e}[/error]")
    
    def learn_topic(self, topic):
        """Interactive tutorial on a specific topic"""
        question = f"""Create an interactive, comprehensive tutorial on "{topic}" for a software engineer.

Structure the tutorial with:
1. **Introduction** - What is it and why is it important?
2. **Core Concepts** - Key ideas explained clearly
3. **Practical Examples** - Working code examples with explanations
4. **Common Patterns** - Best practices and design patterns
5. **Common Pitfalls** - What to avoid and why
6. **Real-world Applications** - Where and how it's used
7. **Next Steps** - What to learn next

Make it engaging, practical, and include code examples that can be run immediately.
Use markdown formatting with syntax-highlighted code blocks."""
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
            transient=True
        ) as progress:
            progress.add_task(f"📚 Creating tutorial for '{topic}'...", total=None)
            answer, metadata = self.ask(question, save_to_history=False)
        
        console.print(Panel(
            Markdown(answer),
            title=f"[cyan]📚 Tutorial: {topic}[/cyan]",
            border_style="cyan",
            padding=(1, 2)
        ))
        
        # Track learning progress
        if "topics" not in self.learning_progress:
            self.learning_progress["topics"] = {}
        self.learning_progress["topics"][topic] = {
            "completed": True,
            "timestamp": datetime.now().isoformat()
        }
        self.save_learning_progress()
        
        console.print(f"\n[success]✅ Tutorial completed! Progress saved.[/success]")
    
    def explain_concept(self, concept):
        """Detailed explanation of a concept"""
        question = f"""Provide a detailed, technical explanation of "{concept}" for software engineers.

Include:
1. **Definition** - Clear, precise definition
2. **How it Works** - Technical details and mechanics
3. **Why it Matters** - Importance and use cases
4. **Code Examples** - Practical implementations
5. **Comparisons** - Compare with related concepts
6. **Best Practices** - How to use it effectively

Be thorough but clear. Include working code examples."""
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
            transient=True
        ) as progress:
            progress.add_task(f"💡 Explaining '{concept}'...", total=None)
            answer, metadata = self.ask(question, save_to_history=False)
        
        console.print(Panel(
            Markdown(answer),
            title=f"[yellow]💡 Explanation: {concept}[/yellow]",
            border_style="yellow",
            padding=(1, 2)
        ))
    
    def explain_like_five(self, concept):
        """ELI5 - Explain Like I'm 5"""
        question = f"""Explain "{concept}" in the simplest way possible, like explaining to a 5-year-old, but keep it technically accurate.

Use:
- Simple analogies and metaphors
- No jargon (or explain jargon simply)
- Real-world comparisons
- Visual descriptions
- Short sentences

Then provide a simple code example that demonstrates the concept clearly.

Make it fun and easy to understand while being technically correct!"""
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
            transient=True
        ) as progress:
            progress.add_task(f"🧒 Creating ELI5 for '{concept}'...", total=None)
            answer, metadata = self.ask(question, save_to_history=False)
        
        console.print(Panel(
            Markdown(answer),
            title=f"[magenta]🧒 ELI5: {concept}[/magenta]",
            border_style="magenta",
            padding=(1, 2)
        ))
    
    def take_quiz(self, topic):
        """Interactive quiz on a topic"""
        question = f"""Create an interactive quiz on "{topic}" with 5 multiple-choice questions.

For each question:
1. State the question clearly
2. Provide 4 options (A, B, C, D)
3. Include one correct answer
4. Make questions progressively harder
5. Cover different aspects of the topic

Format each question like:
**Question 1: [Question text]**
A) [Option A]
B) [Option B]
C) [Option C]
D) [Option D]

After all questions, provide the correct answers with brief explanations.

Make questions practical and relevant to real coding scenarios."""
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
            transient=True
        ) as progress:
            progress.add_task(f"📝 Generating quiz for '{topic}'...", total=None)
            answer, metadata = self.ask(question, save_to_history=False)
        
        console.print(Panel(
            Markdown(answer),
            title=f"[green]📝 Quiz: {topic}[/green]",
            border_style="green",
            padding=(1, 2)
        ))
        
        # Track quiz attempt
        if "quizzes" not in self.learning_progress:
            self.learning_progress["quizzes"] = {}
        
        if topic not in self.learning_progress["quizzes"]:
            self.learning_progress["quizzes"][topic] = []
        
        self.learning_progress["quizzes"][topic].append({
            "timestamp": datetime.now().isoformat(),
            "completed": True
        })
        self.save_learning_progress()
        
        console.print(f"\n[success]✅ Quiz completed! Check answers above.[/success]")
    
    def practice_coding(self, concept):
        """Generate coding exercises"""
        question = f"""Create 3 practical coding exercises for "{concept}" with increasing difficulty.

For each exercise:
1. **Exercise Title** - Clear, descriptive title
2. **Difficulty** - Beginner/Intermediate/Advanced
3. **Problem Description** - What to build/solve
4. **Requirements** - Specific requirements
5. **Example Input/Output** - Test cases
6. **Hints** - Optional hints (hidden)
7. **Solution** - Complete working solution with explanation

Make exercises practical, hands-on, and relevant to real-world coding.
Include edge cases and error handling in solutions."""
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
            transient=True
        ) as progress:
            progress.add_task(f"⚡ Creating exercises for '{concept}'...", total=None)
            answer, metadata = self.ask(question, save_to_history=False)
        
        console.print(Panel(
            Markdown(answer),
            title=f"[blue]⚡ Practice: {concept}[/blue]",
            border_style="blue",
            padding=(1, 2)
        ))
        
        console.print(f"\n[info]💡 Try solving exercises before checking solutions![/info]")
    
    def show_roadmap(self, career_or_tech):
        """Generate learning roadmap"""
        question = f"""Create a comprehensive learning roadmap for "{career_or_tech}".

Structure:
1. **Overview** - What is this career/technology about?
2. **Prerequisites** - What to know before starting
3. **Learning Path** - Step-by-step progression:
   - **Phase 1: Foundations** (0-3 months)
     - Topics to learn
     - Resources
     - Key skills
   - **Phase 2: Intermediate** (3-6 months)
     - Advanced topics
     - Projects to build
     - Skills to master
   - **Phase 3: Advanced** (6-12 months)
     - Expert-level topics
     - Complex projects
     - Specializations
4. **Essential Tools & Technologies** - What to learn
5. **Project Ideas** - Hands-on projects for each phase
6. **Resources** - Books, courses, documentation
7. **Career Path** - Job roles and progression
8. **Common Mistakes to Avoid**

Make it actionable, realistic, and comprehensive."""
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
            transient=True
        ) as progress:
            progress.add_task(f"🗺️  Creating roadmap for '{career_or_tech}'...", total=None)
            answer, metadata = self.ask(question, save_to_history=False)
        
        console.print(Panel(
            Markdown(answer),
            title=f"[cyan]🗺️  Roadmap: {career_or_tech}[/cyan]",
            border_style="cyan",
            padding=(1, 2)
        ))
    
    def show_learning_progress(self):
        """Display learning progress and statistics"""
        topics_completed = len(self.learning_progress.get("topics", {}))
        quizzes_taken = sum(len(q) for q in self.learning_progress.get("quizzes", {}).values())
        
        if topics_completed == 0 and quizzes_taken == 0:
            console.print("[warning]📊 No learning progress yet. Start with 'topic <subject>'![/warning]")
            return
        
        # Topics completed
        console.print("\n[bold cyan]📚 Topics Completed:[/bold cyan]")
        if self.learning_progress.get("topics"):
            topics_table = Table(box=box.ROUNDED, show_header=True, header_style="bold cyan")
            topics_table.add_column("#", style="dim", width=4)
            topics_table.add_column("Topic", style="cyan")
            topics_table.add_column("Completed", style="green")
            
            for i, (topic, data) in enumerate(self.learning_progress["topics"].items(), 1):
                timestamp = datetime.fromisoformat(data["timestamp"]).strftime("%Y-%m-%d %H:%M")
                topics_table.add_row(str(i), topic, timestamp)
            
            console.print(topics_table)
        
        # Quizzes taken
        console.print("\n[bold green]📝 Quizzes Taken:[/bold green]")
        if self.learning_progress.get("quizzes"):
            quiz_table = Table(box=box.ROUNDED, show_header=True, header_style="bold green")
            quiz_table.add_column("Topic", style="green")
            quiz_table.add_column("Attempts", style="yellow", width=10)
            quiz_table.add_column("Last Attempt", style="cyan")
            
            for topic, attempts in self.learning_progress["quizzes"].items():
                count = len(attempts)
                last = datetime.fromisoformat(attempts[-1]["timestamp"]).strftime("%Y-%m-%d %H:%M")
                quiz_table.add_row(topic, str(count), last)
            
            console.print(quiz_table)
        
        # Summary
        summary = f"""
[bold cyan]📊 Learning Summary[/bold cyan]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Topics Completed: {topics_completed}
Quizzes Taken: {quizzes_taken}
Total Activities: {topics_completed + quizzes_taken}

[dim]Keep learning! 🚀[/dim]
        """
        console.print(Panel(summary.strip(), border_style="blue", box=box.ROUNDED))
    
    def show_popular_topics(self):
        """Show popular learning topics"""
        topics = {
            "Programming Fundamentals": [
                "Data Structures", "Algorithms", "Big O Notation", 
                "Recursion", "Object-Oriented Programming"
            ],
            "Web Development": [
                "REST APIs", "Authentication", "Databases", 
                "Frontend Frameworks", "Backend Frameworks"
            ],
            "Design Patterns": [
                "SOLID Principles", "Singleton", "Factory", 
                "Observer", "Dependency Injection"
            ],
            "System Design": [
                "Scalability", "Load Balancing", "Caching", 
                "Microservices", "Message Queues"
            ],
            "DevOps": [
                "Docker", "Kubernetes", "CI/CD", 
                "Monitoring", "Cloud Computing"
            ],
            "Security": [
                "Authentication vs Authorization", "SQL Injection", 
                "XSS", "CSRF", "Encryption"
            ]
        }
        
        console.print(Panel.fit(
            "[bold cyan]📚 Popular Learning Topics[/bold cyan]\n\n"
            "Choose any topic to start learning!",
            border_style="cyan"
        ))
        
        for category, topic_list in topics.items():
            console.print(f"\n[bold yellow]{category}:[/bold yellow]")
            for topic in topic_list:
                console.print(f"  • [cyan]{topic}[/cyan]")
        
        console.print(f"\n[dim]Usage: topic \"<topic_name>\" to start learning[/dim]")
    
    def show_history(self, limit=10):
        """Show recent conversation history"""
        if not self.history:
            console.print("[warning]📝 No history found.[/warning]")
            return
        
        table = Table(title=f"📚 Recent Conversations (last {limit})", 
                     box=box.ROUNDED,
                     show_header=True,
                     header_style="bold cyan")
        table.add_column("#", style="dim", width=4)
        table.add_column("Mode", style="magenta", width=10)
        table.add_column("Timestamp", style="cyan", width=17)
        table.add_column("Question", style="white")
        
        for i, item in enumerate(self.history[-limit:], 1):
            timestamp = datetime.fromisoformat(item["timestamp"]).strftime("%Y-%m-%d %H:%M")
            question = item["question"][:45] + "..." if len(item["question"]) > 45 else item["question"]
            mode = "MULTI" if item.get("mode") == "multi_model" else item.get("provider", "single").upper()[:8]
            table.add_row(str(i), mode, timestamp, question)
        
        console.print(table)
    
    def save_conversation(self, conversation_log, filename):
        """Save current conversation to a file"""
        try:
            filepath = Path(filename)
            mode = "Multi-Model" if self.multi_model_mode else self.current_provider.upper()
            
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write("# DevBot Conversation Log\n")
                f.write(f"# Mode: {mode}\n")
                f.write(f"# Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                
                for i, conv in enumerate(conversation_log, 1):
                    f.write(f"## Question {i}\n{conv['Q']}\n\n")
                    f.write(f"## Answer {i}\n{conv['A']}\n\n")
                    f.write("---\n\n")
            
            console.print(f"[success]✅ Conversation saved to: {filepath.absolute()}[/success]")
        except Exception as e:
            console.print(f"[error]❌ Failed to save: {e}[/error]")
    
    def save_snippet(self, name, content):
        """Save a code snippet for later use"""
        self.snippets[name] = {
            "content": content,
            "timestamp": datetime.now().isoformat()
        }
        self.save_snippets()
        console.print(f"[success]✅ Snippet '{name}' saved![/success]")
    
    def show_snippets(self):
        """Display all saved snippets"""
        if not self.snippets:
            console.print("[warning]📝 No snippets saved yet.[/warning]")
            return
        
        table = Table(title="📎 Saved Code Snippets", 
                     box=box.ROUNDED,
                     show_header=True,
                     header_style="bold magenta")
        table.add_column("Name", style="magenta", width=20)
        table.add_column("Timestamp", style="cyan", width=17)
        table.add_column("Preview", style="white")
        
        for name, data in self.snippets.items():
            timestamp = datetime.fromisoformat(data["timestamp"]).strftime("%Y-%m-%d %H:%M")
            preview = data["content"][:50] + "..." if len(data["content"]) > 50 else data["content"]
            table.add_row(name, timestamp, preview)
        
        console.print(table)
    
    def show_stats(self):
        """Show usage statistics"""
        total_questions = len(self.history)
        total_snippets = len(self.snippets)
        
        # Count by mode
        multi_model_count = len([h for h in self.history if h.get("mode") == "multi_model"])
        single_model_count = total_questions - multi_model_count
        
        if total_questions > 0:
            first_use = datetime.fromisoformat(self.history[0]["timestamp"])
            days_active = (datetime.now() - first_use).days + 1
            avg_per_day = total_questions / days_active
        else:
            days_active = 0
            avg_per_day = 0
        
        stats_text = f"""
[bold cyan]Total Questions Asked:[/bold cyan] {total_questions}
[bold magenta]Saved Snippets:[/bold magenta] {total_snippets}
[bold green]Days Active:[/bold green] {days_active}
[bold yellow]Average Questions/Day:[/bold yellow] {avg_per_day:.1f}

[bold blue]Usage by Mode:[/bold blue]
  • Multi-Model: {multi_model_count}
  • Single Model: {single_model_count}

[bold cyan]Current Mode:[/bold cyan] {"Multi-Model" if self.multi_model_mode else f"Single ({self.current_provider.upper()})"}
        """
        
        console.print(Panel(
            stats_text.strip(),
            title="📊 Usage Statistics",
            border_style="green",
            box=box.DOUBLE
        ))
    
    def show_help(self):
        """Show help information"""
        help_text = """
[bold cyan]DevBot Commands:[/bold cyan]

[bold]/exit[/bold]           - Exit DevBot
[bold]/clear[/bold]          - Clear current chat session
[bold]/history[/bold]        - View recent conversations
[bold]/save[/bold] <file>    - Save conversation to markdown file
[bold]/snippet[/bold] <name> - Save last answer as a snippet
[bold]/snippets[/bold]       - List all saved snippets
[bold]/stats[/bold]          - View usage statistics
[bold]/mode[/bold]           - Toggle multi-model consensus mode
[bold]/provider[/bold]       - Show current provider info
[bold]/api[/bold]            - Enter API testing mode
[bold]/help[/bold]           - Show this help message

[bold cyan]API Testing Commands:[/bold cyan]
get <url>                    - GET request
post <url> <json>            - POST request
put <url> <json>             - PUT request
delete <url>                 - DELETE request
headers <key:value>          - Add header
save <n> <url> <method>   - Save to collection
list                         - List saved APIs
run <n>                   - Run saved API

[bold cyan]File Operations Commands:[/bold cyan]
load <path>                  - Load file into AI context
read <path>                  - Read and display file
write <path> <content>       - Write to file
analyze <path>               - Full code analysis
fix <path> <issue>           - AI fixes code issue
refactor <path>              - AI refactors code
list                         - List loaded files

[bold cyan]Learning Mode Commands:[/bold cyan]
topic <subject>              - Interactive tutorial
explain <concept>            - Detailed explanation
eli5 <concept>               - Explain Like I'm 5
quiz <topic>                 - Take a quiz
practice <concept>           - Coding exercises
roadmap <career>             - Learning roadmap
progress                     - View your progress

[bold cyan]Configuration Commands (CLI):[/bold cyan]
devbot config --set-key <key>       - Set API key
devbot config --provider <name>     - Switch provider
devbot config --multi-model         - Enable multi-model mode
devbot config --show                - Show current config

[bold cyan]🔀 Multi-Model Mode:[/bold cyan]
When enabled, DevBot will:
• Query multiple AI models simultaneously (async)
• Synthesize the best response from all answers
• Combine strengths of Google's intelligence + Groq's speed
• Provide more accurate and comprehensive answers

[bold cyan]Available Providers:[/bold cyan]
• [bold]google[/bold] - Gemini 1.5 Flash (FREE - 1M tokens/day) ⭐
• [bold]groq[/bold] - Ultra fast (300+ tok/s), 14.4K req/day
• [bold]together[/bold] - $25 free credits, great models
• [bold]openrouter[/bold] - Multiple models, flexible

[bold cyan]Tips:[/bold cyan]
• Google Gemini 1.5 Flash is FREE with 1M tokens/day
• Enable multi-model for complex questions
• Use single model for quick queries
• Groq is fastest for real-time responses
• Multi-model gives most comprehensive answers
        """
        
        console.print(Panel(
            help_text.strip(),
            title="❓ Help",
            border_style="blue",
            box=box.ROUNDED
        ))

def main():
    parser = argparse.ArgumentParser(
        description="DevBot - Multi-Model CLI Software Engineering Assistant",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  devbot                                    # Start interactive mode
  devbot ask "How do I reverse a string in Python?"
  devbot config --set-key YOUR_API_KEY     # Set up API key
  devbot config --provider google          # Switch to Google Gemini (FREE)
  devbot config --multi-model              # Enable multi-model mode
  devbot history                           # View recent questions
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # Ask command
    ask_parser = subparsers.add_parser('ask', help='Ask a single question')
    ask_parser.add_argument('question', nargs='+', help='Your question')
    
    # Config command
    config_parser = subparsers.add_parser('config', help='Configure DevBot')
    config_parser.add_argument('--set-key', help='Set API key for current provider')
    config_parser.add_argument('--provider', help='Switch provider (google, groq, together, openrouter)')
    config_parser.add_argument('--multi-model', action='store_true', help='Enable multi-model consensus mode')
    config_parser.add_argument('--single-model', action='store_true', help='Disable multi-model mode')
    config_parser.add_argument('--show', action='store_true', help='Show current configuration')
    
    # History command
    subparsers.add_parser('history', help='View conversation history')
    
    args = parser.parse_args()
    
    # Skip API check when running config command
    skip_check = args.command == 'config'
    bot = DevBot(skip_api_check=skip_check)
    
    if args.command == 'ask':
        question = ' '.join(args.question)
        answer, metadata = bot.ask(question)
        
        title_text = "[bold cyan]🤖 DevBot"
        if metadata.get("mode") == "multi_model":
            providers = metadata.get("providers", [])
            title_text += f" (Synthesized from {', '.join([p.upper() for p in providers])})"
        elif metadata.get("mode") == "single":
            title_text += f" ({metadata.get('provider', '').upper()})"
        title_text += "[/bold cyan]"
        
        console.print(Panel(
            Markdown(answer),
            title=title_text,
            border_style="cyan",
            padding=(1, 2)
        ))
    elif args.command == 'config':
        if args.provider and args.set_key:
            # If both provider and key are specified, set key for that provider
            bot.set_api_key(args.set_key, provider=args.provider)
            # Also switch to that provider
            bot.set_provider(args.provider)
        elif args.set_key:
            bot.set_api_key(args.set_key)
        elif args.provider:
            bot.set_provider(args.provider)
        elif args.multi_model:
            bot.toggle_multi_model(True)
        elif args.single_model:
            bot.toggle_multi_model(False)
        elif args.show:
            mode = "Multi-Model" if bot.multi_model_mode else f"Single ({bot.current_provider})"
            console.print(f"[info]Current Mode:[/info] {mode}")
            console.print(f"[info]API Keys Configured:[/info] {', '.join(bot.config.get('api_keys', {}).keys())}")
            if bot.multi_model_mode:
                providers = bot.config.get('consensus_providers', [])
                console.print(f"[info]Consensus Providers:[/info] {', '.join(providers)}")
        else:
            console.print("[warning]Use --set-key, --provider, --multi-model, or --show[/warning]")
    elif args.command == 'history':
        bot.show_history(20)
    else:
        # Default: interactive mode
        bot.interactive_mode()

if __name__ == "__main__":
    main()