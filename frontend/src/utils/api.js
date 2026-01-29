const API_BASE_URL = "http://localhost:8000";

export const api = {
  async getTransactions(limit = 20, offset = 0, transactionId = null) {
    const params = new URLSearchParams({
      limit: limit.toString(),
      offset: offset.toString(),
    });

    if (transactionId) {
      params.append("transaction_id", transactionId.toString());
    }

    const response = await fetch(`${API_BASE_URL}/transactions?${params}`);
    if (!response.ok) throw new Error("Failed to fetch transactions");
    return response.json();
  },

  async explainTransaction(transactionId, signal = null) {
    const options = {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ transaction_id: transactionId }),
    };

    if (signal) {
      options.signal = signal;
    }

    const response = await fetch(`${API_BASE_URL}/explain`, options);

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || "Failed to generate explanation");
    }

    return response.json();
  },

  async generateNarrative(transactionId, metadata, explanations, pattern = null, trust = null, systemExplanation = null) {
    const response = await fetch(`${API_BASE_URL}/generate-narrative`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        transaction_id: transactionId,
        metadata,
        explanations,
        // Full scorecard for enhanced AI narrative
        pattern,
        trust,
        system_explanation: systemExplanation,
      }),
    });

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || "Failed to generate narrative");
    }

    return response.json();
  },
};
