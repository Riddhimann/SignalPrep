RUBRIC = {
    "relevance": {
        "1-3": "Does not answer the question or is mostly unrelated",
        "4-6": "Addresses the topic but is generic, indirect, or incomplete",
        "7-8": "Directly answers the complete question with role-relevant specifics",
        "9-10": "Precisely answers every part and connects decisions to the role or business need",
    },
    "clarity": {
        "1-3": "Hard to follow, contradictory, or fragmentary",
        "4-6": "Understandable but vague, repetitive, or missing important connections",
        "7-8": "Concise, concrete, and easy to follow",
        "9-10": "Exceptionally precise explanation of complex work for the intended audience",
    },
    "structure": {
        "1-3": "No discernible sequence or answer structure",
        "4-6": "Some sequence, but context, actions, validation, or outcome are missing",
        "7-8": "Clear situation/problem, contribution, method, validation, and result flow",
        "9-10": "Strong narrative hierarchy with explicit trade-offs and learning",
    },
    "technical_depth": {
        "1-3": "Names technology without explaining its use",
        "4-6": "Explains an approach but not key choices, assumptions, or validation",
        "7-8": "Explains implementation decisions, metrics, validation, and trade-offs",
        "9-10": "Deeply reasons about alternatives, failure modes, limitations, and operations",
    },
    "evidence": {
        "definition": "Score evidence stated inside the candidate answer, not the retrieved citation list",
        "1-3": "Unsupported assertion with no example, metric, validation, scale, or outcome",
        "4-6": "Contains one concrete detail but weak validation or outcome evidence",
        "7-8": "Uses specific scale, metrics, validation evidence, and a measurable outcome",
        "9-10": "Uses multiple relevant measurements and explains why they support the conclusion",
    },
}
