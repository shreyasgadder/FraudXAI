import { useState, useEffect } from "react";
import { api } from "../utils/api";

function TransactionTable({ onExplain }) {
    const [transactions, setTransactions] = useState([]);
    const [total, setTotal] = useState(0);
    const [offset, setOffset] = useState(0);
    const [searchId, setSearchId] = useState("");
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);

    const limit = 20;

    const loadTransactions = async () => {
        setLoading(true);
        setError(null);

        try {
            const transactionId = searchId.trim() ? parseInt(searchId) : null;
            const data = await api.getTransactions(limit, offset, transactionId);

            setTransactions(data.transactions);
            setTotal(data.total);
        } catch (err) {
            setError(err.message);
        } finally {
            setLoading(false);
        }
    };

    useEffect(() => {
        loadTransactions();
    }, [offset]);

    const handleSearch = (e) => {
        e.preventDefault();
        setOffset(0);
        loadTransactions();
    };

    const handleReset = () => {
        setSearchId("");
        setOffset(0);
        loadTransactions();
    };

    const getRiskEmoji = (probability) => {
        if (probability >= 0.80) return "🔴";
        if (probability >= 0.60) return "🟡";
        return "🟢";
    };

    const formatCurrency = (amount) => {
        return new Intl.NumberFormat("en-US", {
            style: "currency",
            currency: "USD",
            minimumFractionDigits: 2,
        }).format(amount);
    };

    return (
        <div className="transaction-table-container">
            <div className="table-controls">
                <form onSubmit={handleSearch} className="search-form">
                    <input
                        type="text"
                        placeholder="🔍 Enter Transaction ID..."
                        value={searchId}
                        onChange={(e) => setSearchId(e.target.value)}
                        className="search-input"
                    />
                    <button type="submit" className="btn btn-primary">
                        SEARCH
                    </button>
                    <button type="button" onClick={handleReset} className="btn btn-secondary">
                        RESET
                    </button>
                </form>

                <div className="pagination-info">
                    Showing: {offset + 1}-{Math.min(offset + limit, total)} of {total} Rows
                </div>
            </div>

            {error && <div className="error-message">❌ {error}</div>}

            {loading ? (
                <div className="loading">Loading transactions...</div>
            ) : (
                <>
                    <table className="transaction-table">
                        <thead>
                            <tr>
                                <th>TX_ID</th>
                                <th>HOUR_OF_DAY</th>
                                <th>TYPE</th>
                                <th>AMOUNT ($)</th>
                                <th>SENDER ID</th>
                                <th>RECEIVER ID</th>
                                <th>RISK</th>
                                <th>ACTION</th>
                            </tr>
                        </thead>
                        <tbody>
                            {transactions.map((tx) => (
                                <tr key={tx.transaction_id}>
                                    <td>{tx.transaction_id}</td>
                                    <td>{tx.hour_of_day}</td>
                                    <td>{tx.type}</td>
                                    <td className="amount">{formatCurrency(tx.amount)}</td>
                                    <td className="account-id">{tx.sender_id}</td>
                                    <td className="account-id">{tx.receiver_id}</td>
                                    <td className="risk">
                                        {getRiskEmoji(tx.fraud_probability)} {tx.fraud_probability.toFixed(2)}
                                    </td>
                                    <td>
                                        <button
                                            className="btn btn-explain"
                                            onClick={() => onExplain(tx)}
                                        >
                                            EXPLAIN
                                        </button>
                                    </td>
                                </tr>
                            ))}
                        </tbody>
                    </table>

                    <div className="pagination">
                        <button
                            onClick={() => setOffset(Math.max(0, offset - limit))}
                            disabled={offset === 0}
                        >
                            ← Previous
                        </button>
                        <span className="page-info">
                            Page {Math.floor(offset / limit) + 1} of {Math.ceil(total / limit)}
                        </span>
                        <button
                            onClick={() => setOffset(offset + limit)}
                            disabled={offset + limit >= total}
                        >
                            Next →
                        </button>
                    </div>
                </>
            )}
        </div>
    );
}

export default TransactionTable;
