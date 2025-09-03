rag_prompt = """Jesteś asystentem do odpowiadania na pytania.
Użyj poniższych fragmentów pozyskanego kontekstu, aby odpowiedzieć na pytanie. 
Jeśli nie znasz odpowiedzi, po prostu powiedz, że jej nie znasz. 
Użyj maksymalnie trzech zdań i zachowaj zwięzłość.

Pytanie: {question} 
Kontekst: {context} 
Odpowiedź:"""
