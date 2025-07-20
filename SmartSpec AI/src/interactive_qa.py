def run_query_answering(decoder, encoder, vocab, device):
    """
    Interactive CLI loop to query the document.
    """
    print("\n=== SmartSpec AI Interactive Query Mode ===")
    print("Type your query below (or 'exit' to quit):\n")

    while True:
        query = input(">> ").strip()
        if query.lower() in ["exit", "quit"]:
            print("Exiting interactive query mode.")
            break

        if not query:
            print("Please enter a query or type 'exit'.")
            continue

        # Encode the query (simulated embedding)
        query_ids = [vocab.word_to_id.get(word.lower(), vocab.word_to_id["<unk>"]) for word in query.split()]
        query_tensor = torch.tensor(query_ids, dtype=torch.long, device=device).unsqueeze(1)

        # In production, you'd pass query_tensor through encoder and decoder to generate answer.
        # For this example, we simulate a canned response.
        response = generate_answer(query)

        print(f"\nAI Generated Test Cases:\n{response}\n")


def generate_answer(query_text):
    """
    Simulate answer generation.
    In production, you would:
    - Encode the query
    - Attend over encoder outputs
    - Decode step by step
    """
    # Simulated output
    return (
        "1. Validate user credentials with correct and incorrect inputs.\n"
        "2. Verify password complexity enforcement.\n"
        "3. Confirm account lockout after 3 failed attempts.\n"
        "4. Ensure dashboard refreshes automatically every 5 minutes."
    )
