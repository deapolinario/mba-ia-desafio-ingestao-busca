from search import search_prompt


def main() -> None:
    chain = search_prompt()

    if not chain:
        print("Não foi possível iniciar o chat. Verifique os erros de inicialização.")
        return

    print("Bem-vindo ao chat RAG! Faça perguntas sobre o documento carregado.")
    print("Pressione Ctrl+C para sair.\n")

    while True:
        try:
            print("Faça sua pergunta:")
            question = input("PERGUNTA: ").strip()

            if not question:
                continue

            resposta = chain(question)
            print(f"RESPOSTA: {resposta}\n")

        except KeyboardInterrupt:
            print("\nEncerrando o chat. Até logo!")
            return

        except Exception as e:
            print(f"Erro ao processar pergunta: {e}\n")
            continue


if __name__ == "__main__":
    main()
