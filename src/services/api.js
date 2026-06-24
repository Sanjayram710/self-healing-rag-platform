import axios from "axios";

const api = axios.create({
  baseURL: import.meta.env.VITE_API_URL || "http://127.0.0.1:8000",
  timeout: 120000,
});

let authToken = "";

export const setApiAuthToken = (token) => {
  authToken = token || "";
};

api.interceptors.request.use((config) => {
  if (authToken) {
    config.headers.Authorization = `Bearer ${authToken}`;
  }
  return config;
});

export const askQuestion = async (payload) => {
  const { data } = await api.post("/api/chat", payload);
  return data;
};

export const uploadDocument = async (file, collectionId, onUploadProgress) => {
  const formData = new FormData();
  formData.append("file", file);
  if (collectionId) {
    formData.append("collection_id", collectionId);
  }
  const { data } = await api.post("/api/documents/upload", formData, {
    headers: { "Content-Type": "multipart/form-data" },
    onUploadProgress,
  });
  return data;
};

export const fetchDocuments = async () => {
  const { data } = await api.get("/api/documents");
  return data;
};

export const deleteDocument = async (filename) => {
  const { data } = await api.delete(`/api/documents/${encodeURIComponent(filename)}`);
  return data;
};

export const reindexDocument = async (filename) => {
  const { data } = await api.post(`/api/documents/${encodeURIComponent(filename)}/reindex`);
  return data;
};

export const fetchAnalytics = async (range = "7d", startDate = null, endDate = null) => {
  let url = `/api/analytics?range=${encodeURIComponent(range)}`;
  if (range === "custom" && startDate && endDate) {
    url += `&start_date=${encodeURIComponent(startDate)}&end_date=${encodeURIComponent(endDate)}`;
  }
  console.info("[Analytics Request]", { range, startDate, endDate, url });
  const { data } = await api.get(url);
  return data;
};

export const exportAnalytics = async (format, range = "7d", startDate = null, endDate = null) => {
  let url = `/api/analytics/export?format=${encodeURIComponent(format)}&range=${encodeURIComponent(range)}`;
  if (range === "custom" && startDate && endDate) {
    url += `&start_date=${encodeURIComponent(startDate)}&end_date=${encodeURIComponent(endDate)}`;
  }
  console.info("[Analytics Export Request]", { format, range, startDate, endDate, url });
  const response = await api.get(url, {
    responseType: "blob",
  });
  return response.data; // already a Blob
};


export const streamLogsUrl = `${api.defaults.baseURL}/api/logs/stream`;

export const fetchChats = async () => {
  const { data } = await api.get("/api/chats");
  return data;
};

export const searchChats = async (query = "") => {
  const { data } = await api.get(`/api/chats/search?query=${encodeURIComponent(query)}`);
  return data;
};

export const createChat = async (title, collectionId = null) => {
  const { data } = await api.post("/api/chats", { title, collection_id: collectionId });
  return data;
};

export const fetchChat = async (id) => {
  const { data } = await api.get(`/api/chats/${id}`);
  return data;
};

export const deleteChat = async (id) => {
  const { data } = await api.delete(`/api/chats/${id}`);
  return data;
};

export const updateChat = async (id, updates) => {
  const { data } = await api.put(`/api/chats/${id}`, updates);
  return data;
};

export const fetchSettings = async () => {
  const { data } = await api.get("/api/settings");
  return data;
};

export const updateSettings = async (settings) => {
  const { data } = await api.post("/api/settings", { settings });
  return data;
};

export const fetchCollections = async () => {
  const { data } = await api.get("/api/collections");
  return data;
};

export const createCollection = async (name) => {
  const { data } = await api.post("/api/collections", { name });
  return data;
};

export const deleteCollection = async (id) => {
  const { data } = await api.delete(`/api/collections/${encodeURIComponent(id)}`);
  return data;
};

export default api;
