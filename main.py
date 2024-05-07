import asyncio
import os
import ollama
from datetime import datetime
from colorama import init, Fore, Style
from pyfiglet import Figlet
from termcolor import colored
from chromadb import Client
import itertools
import threading
import sys
import time
import subprocess

init()  # Initialize colorama

def display_loading_animation(status_text, stop_event):
    animation = itertools.cycle(['⣾', '⣽', '⣻', '⢿', '⡿', '⣟', '⣯', '⣷'])
    while not stop_event.is_set():
        sys.stdout.write(Fore.YELLOW + '\r' + next(animation) + ' ' + status_text + Style.RESET_ALL)
        sys.stdout.flush()
        time.sleep(0.2)
    sys.stdout.write('\r' + ' ' * (len(status_text) + 2) + '\r')
    sys.stdout.flush()

def display_ascii_art():
    figlet = Figlet(font='standard')
    ascii_art = figlet.renderText('Chaos AI local assistant')
    colored_ascii_art = colored(ascii_art, 'cyan')
    print(colored_ascii_art)

def display_chat_bubble(message):
    lines = message.split('\n')
    max_line_length = max(len(line) for line in lines)
    
    print(f"{Fore.GREEN}{'_' * (max_line_length + 2)}{Style.RESET_ALL}")
    for line in lines:
        padding = ' ' * (max_line_length - len(line))
        print(f"{Fore.GREEN}| {Fore.YELLOW}{line}{padding} {Fore.GREEN}|{Style.RESET_ALL}")
    print(f"{Fore.GREEN}{'‾' * (max_line_length + 2)}{Style.RESET_ALL}")

def execute_command(command):
    try:
        output = subprocess.check_output(command, shell=True, universal_newlines=True, stderr=subprocess.STDOUT)
        return output, True
    except subprocess.CalledProcessError as e:
        error_message = f"Command '{command}' failed with error:\n{e.output}"
        return error_message, False

async def chat(question, messages, end_word='quit'):
    if question.lower() == end_word:
        return f'{Fore.YELLOW}Conversation ended.{Style.RESET_ALL} ', True

    messages.append({'role': 'user', 'content': question})
    client = ollama.AsyncClient()

    # Display loading animation while generating response
    stop_event = threading.Event()
    loading_thread = threading.Thread(target=display_loading_animation, args=("Generating response...", stop_event))
    loading_thread.start()

    try:
        response = await client.chat(model='llama3', messages=messages, stream=True)
        full_response = ''
        async for chunk in response:
            if 'message' in chunk and 'content' in chunk['message']:
                current_chunk = chunk['message']['content']
                full_response += current_chunk

        stop_event.set()
        loading_thread.join()
        sys.stdout.write(Fore.GREEN + '\rResponse generated!              \n' + Style.RESET_ALL)

        return full_response, False
    except Exception as e:
        stop_event.set()
        loading_thread.join()
        print(f"{Fore.RED}Error while generating response: {str(e)}{Style.RESET_ALL}")
        return "Sorry, an error occurred while generating the response.", False

async def main():
    messages = [{'role': 'system', 'content': 'You are a helpful AI assistant. Respond to the user\'s questions and assist them with their requests.'}]
    model = 'llama3'

    # Load conversation history from file if it exists
    history_file = 'conversation_history.txt'
    if os.path.exists(history_file):
        # Display loading animation while embedding history
        stop_event = threading.Event()
        loading_thread = threading.Thread(target=display_loading_animation, args=("Embedding conversation history...", stop_event))
        loading_thread.start()

        try:
            with open(history_file, 'r', encoding='utf-8') as file:
                for line in file:
                    line = line.strip()
                    if ': ' in line:
                        role, content = line.split(': ', 1)
                        messages.append({'role': role.lower(), 'content': content})
                    else:
                        print(f"{Fore.RED}Invalid line in conversation history: {line}{Style.RESET_ALL}")
                        messages.append({'role': 'user', 'content': line})

            stop_event.set()
            loading_thread.join()
            sys.stdout.write(Fore.GREEN + '\rConversation history embedded!   \n' + Style.RESET_ALL)
        except Exception as e:
            stop_event.set()
            loading_thread.join()
            print(f"{Fore.RED}Error while embedding conversation history: {str(e)}{Style.RESET_ALL}")

    # Initialize ChromaDB client and create a collection
    client = Client()
    collection = client.create_collection(name="docs")

    documents = [
        "Python is a high-level, interpreted programming language known for its simplicity and readability.",
        "Python supports multiple programming paradigms, including object-oriented, imperative, and functional programming.",
        "Python has a vast ecosystem of libraries and frameworks, making it suitable for a wide range of applications.",
        "Python is widely used in web development, data analysis, artificial intelligence, and scientific computing.",
        "Python emphasizes code readability and uses indentation to define code blocks.",
        "Python is an open-source language with a large and active community of developers.",
    ]

    # Display loading animation while storing documents
    stop_event = threading.Event()
    loading_thread = threading.Thread(target=display_loading_animation, args=("Storing documents...", stop_event))
    loading_thread.start()

    # Store each document in the vector embedding database
    try:
        for i, d in enumerate(documents):
            response = ollama.embeddings(model="mxbai-embed-large", prompt=d)
            embedding = response["embedding"]
            collection.add(
                ids=[str(i)],
                embeddings=[embedding],
                documents=[d]
            )

        stop_event.set()
        loading_thread.join()
        sys.stdout.write(Fore.GREEN + '\rDocuments stored!                 \n' + Style.RESET_ALL)
    except Exception as e:
        stop_event.set()
        loading_thread.join()
        print(f"{Fore.RED}Error while storing documents: {str(e)}{Style.RESET_ALL}")

    display_ascii_art()
    print(f"{Fore.BLUE}Welcome to the Chaos AI local assistant! (Type 'quit' to end the conversation){Style.RESET_ALL}")

    while True:
        user_input = input(f"{Fore.YELLOW}You: {Style.RESET_ALL}")
        
        # Display loading animation while retrieving relevant documents
        stop_event = threading.Event()
        loading_thread = threading.Thread(target=display_loading_animation, args=("Retrieving relevant information...", stop_event))
        loading_thread.start()

        try:
            # Generate embeddings for user input and retrieve relevant documents
            response = ollama.embeddings(prompt=user_input, model="mxbai-embed-large")
            if response["embedding"]:
                results = collection.query(
                    query_embeddings=[response["embedding"]],
                    n_results=1
                )
            else:
                results = {'documents': [['']]}  # Set an empty document if embedding is empty

            stop_event.set()
            loading_thread.join()
            sys.stdout.write(Fore.GREEN + '\rRelevant information retrieved!   \n' + Style.RESET_ALL)

            if results['documents'][0][0]:
                data = results['documents'][0][0]
                # Generate response using the retrieved data and user input
                response, conversation_ended = await chat(f"Using this data: {data}. Respond to this prompt: {user_input}", messages)
            else:
                response, conversation_ended = await chat(user_input, messages)
        except Exception as e:
            stop_event.set()
            loading_thread.join()
            print(f"{Fore.RED}Error while retrieving relevant information: {str(e)}{Style.RESET_ALL}")
            response, conversation_ended = await chat(user_input, messages)
        
        # Check if the response contains only a !command
        if response.startswith('!command '):
            command = response[9:].strip()
            
            # Execute the command and get the output
            command_output, success = execute_command(command)
            
            # Display the command output
            print(f"{Fore.CYAN}Command output:{Style.RESET_ALL}")
            print(command_output)
            
            # Generate a new response based on the command success
            if success:
                new_prompt = f"The command '{command}' executed successfully. Here's the output:\n{command_output}\nBased on this, how can I better assist you with your request?"
            else:
                new_prompt = f"The command '{command}' failed with the following error:\n{command_output}\nBased on this, how can I better assist you with your request?"
            
            response, conversation_ended = await chat(new_prompt, messages)
            
            # Save user input, command, command output, and new response to conversation history
            try:
                with open(history_file, 'a', encoding='utf-8') as file:
                    file.write(f"user: {user_input}\n")
                    file.write(f"assistant: {response}\n")
                    file.write(f"command: {command}\n")
                    file.write(f"command_output: {command_output}\n")
                    file.write(f"new_response: {response}\n")
            except Exception as e:
                print(f"{Fore.RED}Error while saving conversation history: {str(e)}{Style.RESET_ALL}")
            
            # Display the new AI response in a chat bubble
            display_chat_bubble(response)
            
            continue
        
        # Save user input and AI response to conversation history
        try:
            with open(history_file, 'a', encoding='utf-8') as file:
                file.write(f"user: {user_input}\n")
                file.write(f"assistant: {response}\n")
        except Exception as e:
            print(f"{Fore.RED}Error while saving conversation history: {str(e)}{Style.RESET_ALL}")

        if conversation_ended:
            break

        # Display the AI's response in a chat bubble
        display_chat_bubble(response)

        # Additional features
        if user_input.lower() == 'change model':
            available_models = ollama.list()['models']
            print(f"{Fore.CYAN}Available models:{Style.RESET_ALL}")
            for i, model_data in enumerate(available_models, start=1):
                print(f"{Fore.CYAN}{i}. {model_data['name']}{Style.RESET_ALL}")
            model_index = int(input(f"{Fore.YELLOW}Enter the number of the model you want to use: {Style.RESET_ALL}"))
            model = available_models[model_index - 1]['name']
            print(f"{Fore.GREEN}Switched to model: {model}{Style.RESET_ALL}")

        elif user_input.lower() == 'show model details':
            model_details = ollama.show(model)
            print(f"{Fore.CYAN}Model details for {model}:{Style.RESET_ALL}")
            for key, value in model_details.items():
                print(f"{Fore.CYAN}{key}: {value}{Style.RESET_ALL}")

        elif user_input.lower() == 'clear history':
            messages = [{'role': 'system', 'content': 'You are a helpful AI assistant. Respond to the user\'s questions and assist them with their requests.'}]
            print(f"{Fore.GREEN}Conversation history cleared.{Style.RESET_ALL}")

        elif user_input.lower() == 'save conversation':
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            save_file = f"conversation_{timestamp}.txt"
            try:
                with open(save_file, 'w', encoding='utf-8') as file:
                    for message in messages:
                        file.write(f"{message['role']}: {message['content']}\n")
                print(f"{Fore.GREEN}Conversation saved to {save_file}{Style.RESET_ALL}")
            except Exception as e:
                print(f"{Fore.RED}Error while saving conversation: {str(e)}{Style.RESET_ALL}")

        elif user_input.lower() == 'help':
            print(f"{Fore.CYAN}Available commands:{Style.RESET_ALL}")
            print(f"{Fore.CYAN}- change model: Change the current model{Style.RESET_ALL}")
            print(f"{Fore.CYAN}- show model details: Display details of the current model{Style.RESET_ALL}")
            print(f"{Fore.CYAN}- clear history: Clear the conversation history{Style.RESET_ALL}")
            print(f"{Fore.CYAN}- save conversation: Save the current conversation to a file{Style.RESET_ALL}")
            print(f"{Fore.CYAN}- help: Show this help message{Style.RESET_ALL}")
            print(f"{Fore.CYAN}- quit: End the conversation{Style.RESET_ALL}")

asyncio.run(main())
