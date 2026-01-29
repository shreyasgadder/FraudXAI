import { useState, useEffect } from "react";
import "./App.css";
import TransactionTable from "./components/TransactionTable";
import InvestigationModal from "./components/InvestigationModal";

function App() {
  const [selectedTransaction, setSelectedTransaction] = useState(null);
  const [isModalOpen, setIsModalOpen] = useState(false);

  return (
    <div className="App">
      <header className="app-header">
        <h1>🔍 FRAUD-XAI 🤖</h1>
        <p>Explainable AI Fraud Detection Dashboard</p>
      </header>

      {/* Full-width transaction table - no container wrapper */}
      <TransactionTable onExplain={setSelectedTransaction} />

      {selectedTransaction && (
        <InvestigationModal
          transaction={selectedTransaction}
          onClose={() => setSelectedTransaction(null)}
        />
      )}
    </div>
  );
}

export default App;
