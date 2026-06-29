import React, { useEffect, useMemo, useRef, useState } from "react";
import {
  BarChart3,
  Bot,
  CheckCircle2,
  Clock,
  FileText,
  Folder,
  HelpCircle,
  Loader2,
  LogOut,
  MessageSquare,
  Pencil,
  Pin,
  PinOff,
  Settings,
  Upload,
  SendHorizontal,
  ShieldCheck,
  Sparkles,
  Trash2,
  Plus,
} from "lucide-react";
import { useAuth } from "../context/AuthContext.jsx";
import { askQuestion, fetchChats, searchChats, createChat, fetchChat, deleteChat, updateChat, fetchCollections, uploadDocument, fetchDocuments } from "../services/api.js";
import StatusPill from "../components/StatusPill.jsx";

export default function ChatPage({ currentChatId, setCurrentChatId, onDocumentUpload }) {
  const chatInputRef = useRef(null);
  const [question, setQuestion] = useState("");
  const [messages, setMessages] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  
  const { user, logout } = useAuth();
  const fileInputRef = React.useRef(null);
  const [uploading, setUploading] = useState(false);
  const [uploadStatus, setUploadStatus] = useState(null); // null | { stage: 'upload'|'index'|'store', fileName: string }
  
  const [chats, setChats] = useState([]);
  const [chatsLoading, setChatsLoading] = useState(false);
  const [chatSearch, setChatSearch] = useState("");
  
  const [collections, setCollections] = useState([]);
  const [selectedCollection, setSelectedCollection] = useState("all");

  const lastResponse = useMemo(() => [...messages].reverse().find((item) => item.type === "answer"), [messages]);
  const exampleQuestions = ["What can you help me with?", "Summarize my documents.", "How does this platform verify answers?"];
  const collectionNames = useMemo(() => {
    return collections.reduce((acc, collection) => {
      acc[collection.id] = collection.name;
      return acc;
    }, {});
  }, [collections]);

  // Dynamic workflow based on loading state and last response
  const workflow = useMemo(() => {
    if (loading) {
      return [
        { label: "Question Received", detail: "Timestamp captured", state: "done" },
        { label: "Vector Retrieval", detail: "ChromaDB semantic search", state: "active" },
        { label: "Answer Gen", detail: "LLM generation step", state: "pending" },
        { label: "Verification Agent", detail: "Checking hallucination status", state: "pending" },
        { label: "Critic Agent", detail: "Context repair if needed", state: "pending" },
      ];
    }
    if (!lastResponse) {
      return [
        { label: "Question Received", detail: "Waiting for input", state: "done" },
        { label: "Vector Retrieval", detail: "ChromaDB semantic search", state: "pending" },
        { label: "Answer Gen", detail: "LLM generation step", state: "pending" },
        { label: "Verification Agent", detail: "Checking hallucination status", state: "pending" },
        { label: "Critic Agent", detail: "Context repair if needed", state: "pending" },
      ];
    }
    const attempts = lastResponse.attempts || 1;
    const isGrounded = lastResponse.grounded;
    return [
      { label: "Question Received", detail: "Timestamp captured", state: "done" },
      { label: "Vector Retrieval", detail: "ChromaDB semantic search", state: "done" },
      { label: "Answer Gen", detail: "LLM generation step", state: "done" },
      { label: "Verification Agent", detail: isGrounded ? "Answer verified" : "Needs review", state: "done" },
      { label: "Critic Agent", detail: attempts > 1 ? "Context repaired after retry" : "No repair needed", state: "done" },
    ];
  }, [loading, lastResponse]);

  const formatChatTimestamp = (timestamp) => {
    if (!timestamp) return "No activity yet";
    const date = new Date(timestamp);
    if (Number.isNaN(date.getTime())) return "No activity yet";
    const dateText = new Intl.DateTimeFormat("en-US", {
      month: "short",
      day: "numeric",
      year: "numeric",
    }).format(date);
    const timeText = new Intl.DateTimeFormat("en-US", {
      hour: "numeric",
      minute: "2-digit",
    }).format(date);
    return `${dateText} - ${timeText}`;
  };

  useEffect(() => {
    const handleShortcut = (event) => {
      if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === "k") {
        event.preventDefault();
        chatInputRef.current?.focus();
      }
    };
    window.addEventListener("keydown", handleShortcut);
    loadCollections();
    loadChats();
    return () => window.removeEventListener("keydown", handleShortcut);
  }, []);

  const loadCollections = async () => {
    try {
      const data = await fetchCollections();
      setCollections(data.collections || []);
    } catch (err) {
      console.error("Failed to load collections", err);
    }
  };

  const loadChats = async () => {
    setChatsLoading(true);
    try {
      const data = chatSearch.trim() ? await searchChats(chatSearch.trim()) : await fetchChats();
      setChats(data.chats || []);
    } catch (err) {
      console.error("Failed to load chats", err);
    } finally {
      setChatsLoading(false);
    }
  };

  useEffect(() => {
    const timeout = window.setTimeout(() => {
      loadChats();
    }, 250);
    return () => window.clearTimeout(timeout);
  }, [chatSearch]);

  useEffect(() => {
    if (!currentChatId) {
      setMessages([]);
      return;
    }
    const loadCurrentChat = async () => {
      try {
        const chat = await fetchChat(currentChatId);
        setMessages(chat.messages || []);
      } catch (err) {
        console.error("Failed to load chat", err);
      }
    };
    loadCurrentChat();
  }, [currentChatId]);

  const handleDeleteChat = async (id) => {
    try {
      await deleteChat(id);
      setChats((current) => current.filter((c) => c.chat_id !== id));
      if (currentChatId === id) {
        setCurrentChatId(null);
      }
    } catch (err) {
      console.error("Failed to delete chat", err);
    }
  };

  const handleRenameChat = async (chat) => {
    const nextTitle = window.prompt("Rename chat", chat.title || "New Chat");
    if (!nextTitle || nextTitle.trim() === chat.title) return;
    try {
      await updateChat(chat.chat_id, { title: nextTitle.trim() });
      setChats((current) => current.map((item) => (
        item.chat_id === chat.chat_id ? { ...item, title: nextTitle.trim(), updated_at: new Date().toISOString() } : item
      )));
    } catch (err) {
      console.error("Failed to rename chat", err);
    }
  };

  const handleTogglePinChat = async (chat) => {
    try {
      const pinned = !chat.pinned;
      await updateChat(chat.chat_id, { pinned });
      setChats((current) => current.map((item) => (
        item.chat_id === chat.chat_id ? { ...item, pinned, updated_at: new Date().toISOString() } : item
      )));
    } catch (err) {
      console.error("Failed to pin chat", err);
    }
  };
  const handleFileChange = async (e) => {
    const files = e.target.files;
    if (!files || files.length === 0) return;
    setUploading(true);
    setError("");
    try {
      for (const file of files) {
        setUploadStatus({ stage: 'upload', fileName: file.name });
        const result = await uploadDocument(file, selectedCollection === "all" ? null : selectedCollection);
        if (result.error) throw new Error(result.error);
        setUploadStatus({ stage: 'store', fileName: file.name });
        // Give a brief moment for the store to settle before refreshing
        await new Promise(resolve => setTimeout(resolve, 600));
      }
      if (onDocumentUpload) {
        await onDocumentUpload();
      }
    } catch (err) {
      console.error('Upload failed', err);
      setError(err.response?.data?.detail || err.message || "Upload failed.");
    } finally {
      setUploading(false);
      setTimeout(() => setUploadStatus(null), 2000);
      e.target.value = "";
    }
  };
  const handleSubmit = async (event) => {
    event.preventDefault();
    const cleanQuestion = question.trim();
    if (!cleanQuestion || loading) return;

    let targetChatId = currentChatId;
    if (!targetChatId) {
       try {
         const collectionId = selectedCollection === "all" ? null : selectedCollection;
         const newChat = await createChat(cleanQuestion.slice(0, 30) + (cleanQuestion.length > 30 ? "..." : ""), collectionId);
         targetChatId = newChat.chat_id;
         setCurrentChatId(targetChatId);
         setChats(curr => [newChat, ...curr]);
       } catch (err) {
         setError("Failed to create new chat session.");
         return;
       }
    }

    setMessages((current) => [...current, { type: "question", text: cleanQuestion, id: crypto.randomUUID() }]);
    setQuestion("");
    setLoading(true);
    setError("");

    try {
      const result = await askQuestion({ 
          question: cleanQuestion, 
          collection_id: selectedCollection === "all" ? null : selectedCollection,
          chat_id: targetChatId 
      });
      setMessages((current) => [
        ...current,
        {
          type: "answer",
          id: crypto.randomUUID(),
          text: result.answer || "No answer was returned.",
          confidence: Number(result.confidence || result.verification_score || 0),
          grounded: Boolean(result.grounded),
          status: result.status || (result.grounded ? "verified" : "insufficient_context"),
          queryType: result.query_type || "Document Question",
          attempts: result.attempts || result.retry_count || 1,
          sources: result.sources || result.source_documents || [],
          searchSource: result.search_source || "Documents",
        },
      ]);
      setChats((current) => {
        const updatedAt = new Date().toISOString();
        const nextChats = current.map((chat) => (
          chat.chat_id === targetChatId
            ? {
                ...chat,
                collection_id: selectedCollection === "all" ? null : selectedCollection,
                updated_at: updatedAt,
                created_at: chat.created_at || updatedAt,
              }
            : chat
        ));
        return nextChats.sort((a, b) => new Date(b.updated_at || 0) - new Date(a.updated_at || 0));
      });
    } catch (apiError) {
      setError(apiError.response?.data?.detail || apiError.message || "The question could not be answered.");
    } finally {
      setLoading(false);
    }
  };

  const today = new Date().toDateString();
  const yesterday = new Date(Date.now() - 86400000).toDateString();
  
  const sortedChats = [...chats].sort((a, b) => {
    if (Boolean(a.pinned) !== Boolean(b.pinned)) return a.pinned ? -1 : 1;
    return new Date(b.updated_at || 0) - new Date(a.updated_at || 0);
  });

  const groupedChats = sortedChats.reduce((acc, chat) => {
      const chatDate = new Date(chat.updated_at).toDateString();
      if (chat.pinned) acc.pinned.push(chat);
      else if (chatDate === today) acc.today.push(chat);
      else if (chatDate === yesterday) acc.yesterday.push(chat);
      else acc.older.push(chat);
      return acc;
  }, { pinned: [], today: [], yesterday: [], older: [] });

  return (
    <div className="relative">
    <main className="grid min-h-[calc(100vh-49px)] grid-cols-1 lg:grid-cols-[280px_minmax(0,1fr)_300px]">
      <aside className="flex flex-col border-b border-line bg-[#FBFAF7] lg:border-b-0 lg:border-r">
        <div className="p-5">
            <button
              type="button"
              onClick={() => setCurrentChatId(null)}
              className="flex w-full items-center justify-center gap-2 rounded-md bg-accent px-4 py-3 text-sm font-black text-white shadow-soft transition hover:bg-[#A93E00]"
            >
              <Plus size={17} /> New Chat
            </button>
            <button
              type="button"
              onClick={() => fileInputRef && fileInputRef.current && fileInputRef.current.click()}
              disabled={uploading}
              className="flex w-full items-center justify-center gap-2 rounded-md bg-accent px-4 py-3 text-sm font-black text-white shadow-soft transition hover:bg-[#A93E00] disabled:opacity-65 mt-2"
            >
              <Upload size={17} />
              {uploading ? 'Uploading...' : 'Upload Document'}
            </button>
            <input
              type="file"
              multiple
              accept=".pdf,.docx,.pptx,.txt,.md,.csv,.xlsx,.jpg,.jpeg,.png"
              ref={fileInputRef}
              style={{ display: 'none' }}
              onChange={handleFileChange}
            />
            {uploadStatus && (
              <div className="mt-3 rounded-md border border-line bg-paper p-3">
                <p className="text-xs font-semibold text-[#6A4034] truncate">{uploadStatus.fileName}</p>
                <div className="mt-2 grid grid-cols-3 gap-1">
                  {['Upload', 'Index', 'Store'].map((label, idx) => {
                    const stageOrder = { upload: 0, index: 1, store: 2 };
                    const currentIdx = stageOrder[uploadStatus.stage] ?? 0;
                    const isDone = idx < currentIdx;
                    const isActive = idx === currentIdx;
                    return (
                      <div
                        key={label}
                        className={`grid h-8 place-items-center text-[10px] font-black uppercase rounded ${
                          isActive ? 'bg-accent text-white' : isDone ? 'bg-[#D9C7BC] text-[#6A4034]' : 'bg-[#FFFEFC] text-muted'
                        }`}
                      >
                        {label}
                      </div>
                    );
                  })}
                </div>
              </div>
            )}
        </div>
        <div className="px-5 pb-3">
          <input
            value={chatSearch}
            onChange={(event) => setChatSearch(event.target.value)}
            className="w-full rounded-md border border-line bg-[#FFFEFC] px-3 py-2 text-xs font-semibold text-[#594A42] outline-none focus:border-accent"
            placeholder="Search chats..."
            aria-label="Search chats"
          />
        </div>
        <div className="flex-1 overflow-y-auto px-5 pb-5 space-y-6">
           {chatsLoading ? (
             <p className="text-sm text-muted">Loading chats...</p>
           ) : chats.length === 0 ? (
             <p className="text-sm text-muted">No recent chats.</p>
           ) : (
             <>
               {groupedChats.pinned.length > 0 && (
                   <div>
                       <p className="mb-2 text-xs font-bold text-[#A18478]">Pinned</p>
                       <div className="space-y-1">
                           {groupedChats.pinned.map((chat) => (
                               <ChatHistoryRow
                                 key={chat.chat_id}
                                 chat={chat}
                                 collectionName={collectionNames[chat.collection_id] || "All Collections"}
                                 isActive={currentChatId === chat.chat_id}
                                 onSelect={() => setCurrentChatId(chat.chat_id)}
                                 onRename={(event) => { event.stopPropagation(); handleRenameChat(chat); }}
                                 onTogglePin={(event) => { event.stopPropagation(); handleTogglePinChat(chat); }}
                                 onDelete={(event) => { event.stopPropagation(); handleDeleteChat(chat.chat_id); }}
                               />
                           ))}
                       </div>
                   </div>
               )}
               {groupedChats.today.length > 0 && (
                   <div>
                       <p className="mb-2 text-xs font-bold text-[#A18478]">Today</p>
                       <div className="space-y-1">
                           {groupedChats.today.map((chat) => (
                               <ChatHistoryRow
                                 key={chat.chat_id}
                                 chat={chat}
                                 collectionName={collectionNames[chat.collection_id] || "All Collections"}
                                 isActive={currentChatId === chat.chat_id}
                                 onSelect={() => setCurrentChatId(chat.chat_id)}
                                 onRename={(event) => { event.stopPropagation(); handleRenameChat(chat); }}
                                 onTogglePin={(event) => { event.stopPropagation(); handleTogglePinChat(chat); }}
                                 onDelete={(event) => { event.stopPropagation(); handleDeleteChat(chat.chat_id); }}
                               />
                           ))}
                       </div>
                   </div>
               )}
               {groupedChats.yesterday.length > 0 && (
                   <div>
                       <p className="mb-2 text-xs font-bold text-[#A18478]">Yesterday</p>
                       <div className="space-y-1">
                           {groupedChats.yesterday.map((chat) => (
                               <ChatHistoryRow
                                 key={chat.chat_id}
                                 chat={chat}
                                 collectionName={collectionNames[chat.collection_id] || "All Collections"}
                                 isActive={currentChatId === chat.chat_id}
                                 onSelect={() => setCurrentChatId(chat.chat_id)}
                                 onRename={(event) => { event.stopPropagation(); handleRenameChat(chat); }}
                                 onTogglePin={(event) => { event.stopPropagation(); handleTogglePinChat(chat); }}
                                 onDelete={(event) => { event.stopPropagation(); handleDeleteChat(chat.chat_id); }}
                               />
                           ))}
                       </div>
                   </div>
               )}
               {groupedChats.older.length > 0 && (
                   <div>
                       <p className="mb-2 text-xs font-bold text-[#A18478]">Older</p>
                       <div className="space-y-1">
                           {groupedChats.older.map((chat) => (
                               <ChatHistoryRow
                                 key={chat.chat_id}
                                 chat={chat}
                                 collectionName={collectionNames[chat.collection_id] || "All Collections"}
                                 isActive={currentChatId === chat.chat_id}
                                 onSelect={() => setCurrentChatId(chat.chat_id)}
                                 onRename={(event) => { event.stopPropagation(); handleRenameChat(chat); }}
                                 onTogglePin={(event) => { event.stopPropagation(); handleTogglePinChat(chat); }}
                                 onDelete={(event) => { event.stopPropagation(); handleDeleteChat(chat.chat_id); }}
                               />
                           ))}
                       </div>
                   </div>
               )}
             </>
           )}
        </div>
        <div className="border-t border-line p-5 mt-auto">
          <div className="flex items-center justify-between gap-3">
             <div className="flex items-center gap-3 min-w-0">
               {user?.profile_picture ? (
                  <img src={user.profile_picture} alt="Profile" className="h-8 w-8 rounded-full object-cover shrink-0" />
               ) : (
                  <div className="grid h-8 w-8 place-items-center rounded-full bg-[#FFE0D3] text-xs font-black text-accent shrink-0">
                    {(user?.name || "U").slice(0, 1).toUpperCase()}
                  </div>
               )}
               <div className="min-w-0">
                 <p className="truncate text-sm font-black text-ink">{user?.name || "Authenticated User"}</p>
                 <p className="truncate text-[11px] font-semibold text-muted">{user?.email || ""}</p>
               </div>
             </div>
             <button type="button" onClick={logout} className="shrink-0 text-muted hover:text-red-600 transition" aria-label="Logout">
                <LogOut size={16} />
             </button>
          </div>
        </div>
      </aside>

      <section className="flex min-h-[640px] flex-col border-b border-line lg:border-b-0">
        <div className="scrollbar-soft flex-1 overflow-y-auto px-5 py-6 sm:px-8">
          {messages.length === 0 && (
            <div className="mx-auto mt-12 max-w-xl text-center">
              <ShieldCheck className="mx-auto mb-4 text-accent" size={36} />
              <h1 className="text-3xl font-black">Ask your knowledge base</h1>
              <p className="mt-3 text-sm leading-6 text-muted">
                Answers are checked for grounding against retrieved context before they are returned.
              </p>
              <div className="mt-7 grid gap-2 text-left">
                {exampleQuestions.map((example) => (
                  <button
                    key={example}
                    type="button"
                    onClick={() => {
                      setQuestion(example);
                      chatInputRef.current?.focus();
                    }}
                    className="rounded-lg border border-line bg-[#FFFEFC] px-4 py-3 text-sm font-semibold text-[#6A4034] shadow-soft transition hover:border-accent hover:text-accent"
                  >
                    {example}
                  </button>
                ))}
              </div>
            </div>
          )}
          <div className="mx-auto flex max-w-2xl flex-col gap-6">
            {messages.map((message) =>
              message.type === "question" ? (
                <div key={message.id} className="ml-auto max-w-[80%] rounded-2xl bg-[#E9E5E1] px-5 py-4 text-sm font-medium">
                  {message.text}
                </div>
              ) : (
                <article key={message.id} className="overflow-hidden rounded-xl border border-line bg-[#FFFEFC] shadow-soft">
                  <div className="flex flex-wrap items-center gap-4 bg-[#F5F1ED] px-5 py-3 text-xs font-semibold">
                    <StatusPill status={message.grounded ? "verified" : "processing"}>
                      {message.grounded ? "Verified" : "Needs Review"}
                    </StatusPill>
                    <span className="flex items-center gap-1 rounded bg-[#E4DCD5] px-2 py-0.5 text-[#594A42]">
                      Source: {message.searchSource || "Documents"}
                    </span>
                    <span>Confidence: {message.confidence}%</span>
                    <span>Attempts: {message.attempts}</span>
                  </div>
                  <div className="px-5 py-5">
                    <p className="whitespace-pre-wrap text-sm leading-7 text-[#38302C]">{message.text}</p>
                    {message.sources && message.sources.length > 0 && (
                      <div className="mt-3 border-t border-line pt-2.5">
                        <p className="mb-1.5 text-[10px] font-bold uppercase tracking-[0.15em] text-[#A18478]">Sources</p>
                        <div className="grid grid-cols-1 gap-1.5 sm:grid-cols-2">
                          {message.sources.map((source, index) => (
                            <div
                              key={`${source.filename || source.name || source.source || index}`}
                              className="flex min-h-0 max-h-[64px] items-center gap-2 overflow-hidden rounded border border-line bg-[#F9F6F3] px-2.5 py-1.5"
                              title={source.filename || source.name || source.source || `Source ${index + 1}`}
                            >
                              <FileText size={13} className="shrink-0 text-[#A18478]" />
                              <div className="min-w-0 flex-1">
                                <p className="truncate text-[11px] font-bold leading-tight text-[#38302C]">{source.filename || source.name || source.source || `Source ${index + 1}`}</p>
                                <p className="mt-0.5 truncate text-[10px] leading-tight text-muted">{source.page ? `pg. ${source.page}` : "Retrieved context"}</p>
                              </div>
                            </div>
                          ))}
                        </div>
                      </div>
                    )}
                  </div>
                </article>
              )
            )}
            {loading && (
              <div className="flex items-center gap-2 text-sm font-semibold text-accent">
                <Loader2 className="animate-spin" size={17} />
                Running self-healing workflow...
              </div>
            )}
            {error && <p className="rounded-lg bg-red-50 p-4 text-sm text-red-700">{error}</p>}
          </div>
        </div>
        <form onSubmit={handleSubmit} className="border-t border-line bg-[#FBFAF7] p-5">
          <div className="mx-auto max-w-2xl rounded-xl border border-line bg-[#F1EEEB] p-2 shadow-soft flex items-center gap-2">
            <select
              value={selectedCollection}
              onChange={(event) => setSelectedCollection(event.target.value)}
              className="w-32 rounded-md border border-line bg-[#FFFEFC] px-2 py-2 text-xs font-bold text-[#6A4034] outline-none"
              aria-label="Collection scope"
            >
              <option value="all">All Collections</option>
              {collections.map((col) => (
                <option key={col.id} value={col.id}>{col.name}</option>
              ))}
            </select>
            <div className="h-8 w-px bg-line"></div>
            <div className="flex-1 flex items-center gap-3">
              <input
                ref={chatInputRef}
                value={question}
                onChange={(event) => setQuestion(event.target.value)}
                className="min-w-0 flex-1 bg-transparent px-3 py-4 text-sm outline-none placeholder:text-[#766760]"
                placeholder="Ask a question about your documents..."
              />
              <button type="submit" disabled={loading} className="grid h-10 w-10 shrink-0 place-items-center rounded-md bg-accent text-white disabled:opacity-60">
                {loading ? <Loader2 className="animate-spin" size={18} /> : <SendHorizontal size={19} />}
              </button>
            </div>
          </div>
          <div className="mx-auto mt-3 flex max-w-2xl gap-5 text-xs font-semibold text-[#6A4034]">
            <span className="flex items-center gap-1"><Sparkles size={13} /> Self-Healing Enabled</span>
          </div>
        </form>
      </section>

      <aside className="bg-[#FBFAF7] p-6 lg:border-l lg:border-line">
        <p className="text-sm font-semibold uppercase tracking-[0.2em] text-[#A18478]">Live Workflow</p>
        <p className="mt-2 text-xs font-semibold text-[#6A4034]">Self-Healing Cycle Logs</p>
        <div className="mt-8 space-y-5">
          {workflow.map((step, index) => (
            <div key={step.label} className="flex gap-4">
              <div className="flex flex-col items-center">
                <span className={`grid h-7 w-7 place-items-center rounded-full border ${step.state === "active" ? "border-accent bg-accent text-white" : "border-line bg-paper text-muted"}`}>
                  {step.state === "done" ? <CheckCircle2 size={14} /> : <Bot size={14} />}
                </span>
                {index < workflow.length - 1 && <span className="h-8 w-px bg-line" />}
              </div>
              <div>
                <p className={`text-sm font-black ${step.state === "active" ? "text-accent" : "text-ink"}`}>{step.label}</p>
                <p className="text-[11px] text-muted">{step.detail}</p>
              </div>
            </div>
          ))}
        </div>
        <div className="mt-7 rounded-md bg-paper p-4">
          <p className="text-[11px] font-bold uppercase text-[#8E6B5E]">Grounded Check</p>
          <div className="mt-4 flex items-center justify-between text-xs font-black">
            <span className={lastResponse?.grounded === false ? "text-red-600" : "text-[#D9C7BC]"}>Retry</span>
            <span className="h-px flex-1 bg-line mx-4" />
            <span className={lastResponse?.grounded ? "text-green-700" : "text-[#D9C7BC]"}>Answer</span>
          </div>
        </div>
      </aside>
    </main>
    </div>
  );
}

function ChatHistoryRow({ chat, collectionName, isActive, onSelect, onRename, onTogglePin, onDelete }) {
  return (
    <div
      className={`group relative cursor-pointer rounded-md p-3 pr-24 transition ${isActive ? "border border-line bg-paper" : "hover:bg-paper"}`}
      onClick={onSelect}
      role="button"
      tabIndex={0}
      onKeyDown={(event) => {
        if (event.key === "Enter" || event.key === " ") {
          event.preventDefault();
          onSelect();
        }
      }}
    >
      <div className="flex min-w-0 items-start gap-3">
        <MessageSquare size={16} className={`mt-0.5 shrink-0 ${isActive ? "text-accent" : "text-muted"}`} />
        <div className="min-w-0 flex-1">
          <p className={`truncate text-sm font-semibold ${isActive ? "text-ink" : "text-[#594A42]"}`}>{chat.title}</p>
          <p className="mt-1 flex min-w-0 items-center gap-1 truncate text-[11px] font-semibold text-[#6A4034]"><Folder size={11} className="shrink-0" /><span className="truncate">{collectionName}</span></p>
          <p className="mt-0.5 flex min-w-0 items-center gap-1 truncate text-[11px] font-semibold text-muted"><Clock size={11} className="shrink-0" /><span className="truncate">{formatChatTimestamp(chat.updated_at || chat.created_at)}</span></p>
        </div>
      </div>
      <div className="absolute right-3 top-3 flex items-center gap-2 opacity-0 transition group-hover:opacity-100 focus-within:opacity-100">
        <button
          type="button"
          onClick={onTogglePin}
          className="text-muted transition hover:text-accent"
          aria-label={chat.pinned ? `Unpin ${chat.title}` : `Pin ${chat.title}`}
        >
          {chat.pinned ? <PinOff size={14} /> : <Pin size={14} />}
        </button>
        <button
          type="button"
          onClick={onRename}
          className="text-muted transition hover:text-accent"
          aria-label={`Rename ${chat.title}`}
        >
          <Pencil size={14} />
        </button>
        <button
          type="button"
          onClick={onDelete}
          className="text-muted transition hover:text-red-600"
          aria-label={`Delete ${chat.title}`}
        >
          <Trash2 size={14} />
        </button>
      </div>
    </div>
  );
}


