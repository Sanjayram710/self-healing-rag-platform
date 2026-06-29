import React, { useEffect, useMemo, useState } from "react";
import { BarChart3, FileText, HelpCircle, Loader2, LogOut, MessageSquare, Settings } from "lucide-react";
import ChatPage from "./pages/ChatPage.jsx";
import DocumentsPage from "./pages/DocumentsPage.jsx";
import AnalyticsPage from "./pages/AnalyticsPage.jsx";
import SettingsPage from "./pages/SettingsPage.jsx";
import LoginPage from "./pages/LoginPage.jsx";
import { useAuth } from "./context/AuthContext.jsx";
import { fetchDocuments, setApiAuthToken } from "./services/api.js";

const navItems = [
  { id: "chat", label: "Chat", icon: MessageSquare },
  { id: "documents", label: "Documents", icon: FileText },
  { id: "analytics", label: "Analytics", icon: BarChart3 },
];

export default function App() {
  const { user, token, loading: authLoading, error: authError, isAuthenticated, loginWithGoogle, logout } = useAuth();
  const [activePage, setActivePage] = useState("chat");
  const [documents, setDocuments] = useState([]);
  const [documentsLoading, setDocumentsLoading] = useState(false);
  const [documentsError, setDocumentsError] = useState("");
  const [currentChatId, setCurrentChatId] = useState(null);

  useEffect(() => {
    setApiAuthToken(token);
  }, [token]);

  const loadDocuments = async () => {
    if (!isAuthenticated) return;

    setDocumentsLoading(true);
    setDocumentsError("");
    try {
      const data = await fetchDocuments();
      setDocuments(data.documents || []);
    } catch (error) {
      setDocumentsError(error.response?.data?.detail || error.message || "Unable to load documents.");
    } finally {
      setDocumentsLoading(false);
    }
  };

  useEffect(() => {
    if (isAuthenticated) {
      loadDocuments();
      return;
    }

    setDocuments([]);
    setDocumentsError("");
    setDocumentsLoading(false);
  }, [isAuthenticated]);

  const page = useMemo(() => {
    if (activePage === "documents") {
      return (
        <DocumentsPage
          documents={documents}
          loading={documentsLoading}
          error={documentsError}
          onRefresh={loadDocuments}
        />
      );
    }
    if (activePage === "analytics") {
      return <AnalyticsPage documentsCount={documents.length} />;
    }
    if (activePage === "settings") {
      return <SettingsPage />;
    }
    return (
      <ChatPage
        currentChatId={currentChatId}
        setCurrentChatId={setCurrentChatId}
        onDocumentUpload={loadDocuments}
      />
    );
  }, [activePage, currentChatId, setCurrentChatId]);

  if (authLoading && !isAuthenticated) {
    return (
      <div className="grid min-h-screen place-items-center bg-ivory text-accent">
        <div className="flex items-center gap-3 rounded-xl border border-line bg-paper px-6 py-4 text-sm font-black shadow-soft">
          <Loader2 className="animate-spin" size={18} />
          Restoring secure workspace...
        </div>
      </div>
    );
  }

  if (!isAuthenticated) {
    return <LoginPage onLogin={loginWithGoogle} loading={authLoading} error={authError} />;
  }

  return (
    <div className="min-h-screen bg-ivory text-ink">
      <header className="sticky top-0 z-30 border-b border-line bg-ivory/95 backdrop-blur">
        <div className="flex min-h-12 items-center justify-between px-4 sm:px-7">
          <div className="flex min-w-0 items-center gap-5">
            <button
              type="button"
              onClick={() => setActivePage("chat")}
              className="truncate text-left text-xl font-black text-accent sm:text-2xl"
            >
              Self-Healing RAG Platform
            </button>
            <p className="hidden border-l border-line pl-4 text-xs text-muted lg:block">
              Intelligent Verification & Automated Context Repair
            </p>
          </div>
          <nav className="hidden items-center gap-5 md:flex">
            {navItems.map((item) => (
              <button
                key={item.id}
                type="button"
                onClick={() => setActivePage(item.id)}
                className={`border-b-2 px-1 py-4 text-sm font-semibold transition ${
                  activePage === item.id
                    ? "border-accent text-accent"
                    : "border-transparent text-[#594741] hover:text-accent"
                }`}
              >
                {item.label}
              </button>
            ))}
          </nav>
          <div className="flex items-center gap-2">
            <button type="button" className="icon-button" aria-label="Help">
              <HelpCircle size={18} />
            </button>
            <button type="button" onClick={() => setActivePage("settings")} className="icon-button" aria-label="Settings">
              <Settings size={18} />
            </button>
            <button type="button" onClick={logout} className="icon-button sm:hidden" aria-label="Logout">
              <LogOut size={17} />
            </button>
          </div>
        </div>
        <nav className="grid grid-cols-3 border-t border-line md:hidden">
          {navItems.map((item) => {
            const Icon = item.icon;
            return (
              <button
                key={item.id}
                type="button"
                onClick={() => setActivePage(item.id)}
                className={`flex items-center justify-center gap-2 border-b-2 py-3 text-xs font-bold ${
                  activePage === item.id ? "border-accent text-accent" : "border-transparent text-muted"
                }`}
              >
                <Icon size={15} />
                {item.label}
              </button>
            );
          })}
        </nav>
      </header>
      {page}
    </div>
  );
}
