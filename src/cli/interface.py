#!/usr/bin/env python3
"""Command-line interface for the CHT Documentation Q&A Chatbot."""


import asyncio
from rich.console import Console
from rich.markdown import Markdown
from rich.prompt import Prompt
from rich.table import Table
from core.rag_chain import RAGChain
from utils import load_config


class ChatCLI:
    """Command-line interface for the chatbot."""

    def __init__(self):
        """Initialize the CLI interface."""
        self.console = Console()
        self.rag_chain = RAGChain()

        # Configure styling
        self.user_style = "bold cyan"
        self.assistant_style = "bold green"
        self.source_style = "dim"

    def print_welcome(self):
        """Print welcome message and instructions."""
        welcome_text = """
            # CHT Documentation Assistant

            Welcome to the Community Health Toolkit Documentation Assistant! 
            I can help you find information and answer questions about CHT.

            ## Commands:
            - Type your question and press Enter
            - Type 'clear' to clear the conversation history
            - Type 'exit' or 'quit' to end the session

            ## Tips:
            - Be specific in your questions
            - I'll provide source links for my answers
            - You can ask follow-up questions
        """
        self.console.print(Markdown(welcome_text))
        self.console.print("\n")

    def print_sources(self, sources):
        """Print source information in a table.

        Args:
            sources: List of source dictionaries.
        """
        if not sources:
            return

        table = Table(title="Sources", show_header=True, header_style="bold magenta")
        table.add_column("Title", style="cyan")
        table.add_column("URL", style="blue")
        table.add_column("Relevance", justify="right", style="green")

        for source in sources:
            table.add_row(source["title"], source["url"], f"{source['score']:.2f}")

        self.console.print(table)
        self.console.print("\n")

    def format_answer(self, answer: str) -> str:
        """Format the answer text for display.

        Args:
            answer: Raw answer text.

        Returns:
            Formatted answer text.
        """
        # Add markdown formatting if not present
        if not answer.startswith("#") and not answer.startswith(">"):
            answer = f"> {answer}"

        return answer

    async def handle_question(self, question: str):
        """Handle a user question.

        Args:
            question: User's question text.
        """
        # Show thinking indicator
        with self.console.status("[bold yellow]Thinking...", spinner="dots"):
            try:
                response = await self.rag_chain.answer_question(question)

                # Print the answer
                formatted_answer = self.format_answer(response["answer"])
                self.console.print(
                    Markdown(formatted_answer), style=self.assistant_style
                )
                self.console.print("\n")

                # Print sources
                self.print_sources(response["sources"])

            except Exception as e:
                self.console.print(f"[bold red]Error:[/] {str(e)}")

    async def run(self):
        """Run the CLI interface."""
        try:
            # Load config to verify API keys
            load_config()

            self.print_welcome()

            while True:
                # Get user input
                question = Prompt.ask("[bold cyan]You")

                # Handle commands
                if question.lower() in ["exit", "quit"]:
                    self.console.print("[yellow]Goodbye![/]")
                    break
                elif question.lower() == "clear":
                    self.rag_chain.clear_history()
                    self.console.clear()
                    self.print_welcome()
                    continue

                # Handle question
                await self.handle_question(question)

        except KeyboardInterrupt:
            self.console.print("\n[yellow]Goodbye![/]")
        except Exception as e:
            self.console.print(f"[bold red]Fatal Error:[/] {str(e)}")
            sys.exit(1)


def main():
    """Main entry point."""
    cli = ChatCLI()
    asyncio.run(cli.run())


if __name__ == "__main__":
    main()
