document.addEventListener("DOMContentLoaded", () => {
   // ----------------- Element Selections -----------------
  const searchInput = document.getElementById("searchInput");
  const searchButton = document.getElementById("searchButton");
  const heroSection = document.getElementById("heroSection");
  const chatFlow = document.getElementById("chatFlow");
  const conversationHistory = document.getElementById("conversationHistory");
  const quickHelpContainer = document.getElementById("quickHelpContainer");
  
  const chatInput = document.getElementById("chatInput");
  const chatSendButton = document.getElementById("chatSendButton");
  const relatedQuestionsSection = document.getElementById(
    "relatedQuestionsSection"
  );
  const relatedQuestionsList = document.getElementById("relatedQuestionsList");

    // ----------------- State Variables -----------------
  let suggestionTimeout;
  let currentSuggestions = [];
  let selectedSuggestionIndex = -1;
  let suggestionsVisible = false;
  let virtualList = null;
  let messageCounter = 0;
  let chatSuggestionsVisible = false;
  let chatSelectedSuggestionIndex = -1;
  let currentChatSuggestions = [];
  let chatVirtualList = null;
  let isAnswerLoading = false;
  let currentAbortController = null;
  let currentLanguage = "EN";
// ----------------- Language Management -----------------
/**
 * Switch between English and German languages
 * @param {string} lang - Language code ('EN' or 'DE')
 */
function switchLanguage(lang) {
    currentLanguage = lang;
    
    // Update UI text elements
    const elementsToUpdate = [
        { selector: '.hero-title', attr: `data-${lang.toLowerCase()}` },
        { selector: '.logo-text', attr: `data-${lang.toLowerCase()}` },
        { element: searchInput, attr: `data-placeholder-${lang.toLowerCase()}` },
        { element: chatInput, attr: `data-placeholder-${lang.toLowerCase()}` }
    ];

    elementsToUpdate.forEach(item => {
        if (item.element) {
            item.element.placeholder = item.element.getAttribute(item.attr) || '';
        } else {
          const el = document.querySelector(item.selector);
          if (el) el.textContent = el.getAttribute(item.attr) || '';
        }
    }); 
    
    // Clear suggestions and reload quick help
    hideSuggestions();
    hideChatSuggestions();
    
    // Language-specific loading message
    const loadingText = currentLanguage === 'DE' 
        ? 'Lade Vorschläge...' 
        : 'Loading suggestions...';
    quickHelpContainer.innerHTML = `
        <div id="quickHelpLoading" class="loading-state">
            <div class="loading-spinner"></div>
            <p class="loading-text">${loadingText}</p>
        </div>
    `;
    
    loadQuickHelpSuggestions();
    
    // Clear search input if any
    searchInput.value = '';
}

    // ----------------- Marked.js Configuration -----------------
  marked.setOptions({
    highlight: function (code, lang) {
      if (lang && hljs.getLanguage(lang)) {
        return hljs.highlight(code, { language: lang }).value;
      }
      return hljs.highlightAuto(code).value;
    },
    breaks: true,
    gfm: true,
  });

    // ----------------- Cache Management -----------------
    /**
     * IndexedDB Cache Manager for storing suggestions
     */
  class SuggestionCache {
    constructor() {
      this.dbName = "Arena2036SuggestionsDB";
      this.dbVersion = 1;
      this.storeName = "suggestions";
      this.db = null;
      this.memoryCache = new Map();
      this.maxMemorySize = 500;
      this.cacheExpiry = 24 * 60 * 60 * 1000; // 24 hours
      this.init();
    }

    async init() {
      try {
        this.db = await this.openDB();
        console.log("IndexedDB initialized successfully");
      } catch (error) {
        console.warn("IndexedDB failed to initialize:", error);
      }
    }

    openDB() {
      return new Promise((resolve, reject) => {
        const request = indexedDB.open(this.dbName, this.dbVersion);

        request.onerror = () => reject(request.error);
        request.onsuccess = () => resolve(request.result);

        request.onupgradeneeded = (event) => {
          const db = event.target.result;
          if (!db.objectStoreNames.contains(this.storeName)) {
            const store = db.createObjectStore(this.storeName, {
              keyPath: "query",
            });
            store.createIndex("timestamp", "timestamp", { unique: false });
          }
        };
      });
    }

    async get(query, language = "EN") {
      const cacheKey = `${language}_${query.toLowerCase().trim()}`;

      // Check memory cache first
      if (this.memoryCache.has(cacheKey)) {
        const cached = this.memoryCache.get(cacheKey);
        if (Date.now() - cached.timestamp < this.cacheExpiry) {
          return cached.suggestions;
        } else {
          this.memoryCache.delete(cacheKey);
        }
      }

      // Check IndexedDB
      if (!this.db) return null;

      try {
        const transaction = this.db.transaction([this.storeName], "readonly");
        const store = transaction.objectStore(this.storeName);

        return new Promise((resolve) => {
          const request = store.get(cacheKey);

          request.onsuccess = () => {
            const result = request.result;
            if (result && Date.now() - result.timestamp < this.cacheExpiry) {
              // Update memory cache
              this.memoryCache.set(cacheKey, {
                suggestions: result.suggestions,
                timestamp: result.timestamp,
              });
              resolve(result.suggestions);
            } else {
              resolve(null);
            }
          };

          request.onerror = () => resolve(null);
        });
      } catch (error) {
        console.warn("Cache get error:", error);
        return null;
      }
    }

    async set(query, suggestions, language = "EN") {
      const cacheKey = `${language}_${query.toLowerCase().trim()}`;
      const timestamp = Date.now();
      const cacheData = { suggestions, timestamp };

      // Update memory cache
      this.memoryCache.set(cacheKey, cacheData);

      // Limit memory cache size
      if (this.memoryCache.size > this.maxMemorySize) {
        const firstKey = this.memoryCache.keys().next().value;
        this.memoryCache.delete(firstKey);
      }

      // Update IndexedDB
      if (!this.db) return;

      try {
        const transaction = this.db.transaction([this.storeName], "readwrite");
        const store = transaction.objectStore(this.storeName);

        store.put({
          query: cacheKey,
          suggestions,
          timestamp,
        });
      } catch (error) {
        console.warn("Cache set error:", error);
      }
    }

    async clear() {
      this.memoryCache.clear();
      if (!this.db) return;

      try {
        const transaction = this.db.transaction([this.storeName], "readwrite");
        const store = transaction.objectStore(this.storeName);
        store.clear();
      } catch (error) {
        console.warn("Cache clear error:", error);
      }
    }
  }

    // ----------------- Virtual Scrolling Implementation -----------------
    /**
     * Virtual list for efficient rendering of large suggestion lists
     */
  class VirtualSuggestionsList {
    constructor(container, inputElement = null) {
      this.container = container;
      this.inputElement = inputElement; // ADD THIS
      this.itemHeight = 44;
      this.visibleCount = Math.min(8, Math.floor(300 / this.itemHeight));
      this.scrollTop = 0;
      this.data = [];
      this.selectedIndex = -1;
    }

    render(suggestions, selectedIndex = -1) {
      this.data = suggestions;
      this.selectedIndex = selectedIndex;

      if (suggestions.length === 0) {
        this.container.innerHTML =
           `<div class="no-suggestions">${
                currentLanguage === 'DE' ? 'Keine Vorschläge gefunden' : 'No suggestions found'
            }</div>`;
        return;
      }

      if (suggestions.length <= 10) {
        this.renderDirect(suggestions, selectedIndex);
        return;
      }

      const totalHeight = suggestions.length * this.itemHeight;
      const startIndex = Math.floor(this.scrollTop / this.itemHeight);
      const endIndex = Math.min(
        startIndex + this.visibleCount + 2,
        suggestions.length
      );

      const visibleItems = suggestions.slice(startIndex, endIndex);

      this.container.innerHTML = `
                <div class="virtual-container" style="height: ${totalHeight}px; position: relative;">
                    <div class="virtual-content" style="transform: translateY(${
                      startIndex * this.itemHeight
                    }px);">
                        ${visibleItems
                          .map((suggestion, index) => {
                            const actualIndex = startIndex + index;
                            const isSelected = actualIndex === selectedIndex;
                            return `<div class="suggestion-item ${
                              isSelected ? "selected" : ""
                            }" 
                                        data-index="${actualIndex}" 
                                        style="height: ${
                                          this.itemHeight
                                        }px; line-height: ${
                              this.itemHeight
                            }px;">
                                        <span class="suggestion-icon">🔍</span>
                                        <span class="suggestion-text">${this.highlightMatch(
                                          suggestion,
                                          this.inputElement ? this.inputElement.value : "" // CHANGE THIS
                                        )}</span>
                                    </div>`;
                          })
                          .join("")}
                    </div>
                </div>
            `;
    }

    renderDirect(suggestions, selectedIndex = -1) {
      this.container.innerHTML = suggestions
        .map((suggestion, index) => {
          const isSelected = index === selectedIndex;
          return `<div class="suggestion-item ${isSelected ? "selected" : ""}" 
                            data-index="${index}">
                            <span class="suggestion-icon">🔍</span>
                            <span class="suggestion-text">${this.highlightMatch(
                              suggestion,
                              this.inputElement ? this.inputElement.value : "" // CHANGE THIS
                            )}</span>
                        </div>`;
        })
        .join("");
    }

    highlightMatch(text, query) {
      if (!query || query.length < 2) return text;

      const regex = new RegExp(
        `(${query.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")})`,
        "gi"
      );
      return text.replace(regex, "<mark>$1</mark>");
    }

    handleScroll(scrollTop) {
      if (this.data.length <= 10) return;

      this.scrollTop = scrollTop;
      this.render(this.data, this.selectedIndex);
    }

    updateSelection(newIndex) {
      this.selectedIndex = newIndex;
      this.render(this.data, newIndex);

      if (this.data.length > 10) {
        const itemTop = newIndex * this.itemHeight;
        const itemBottom = itemTop + this.itemHeight;
        const containerTop = this.scrollTop;
        const containerBottom =
          containerTop + this.visibleCount * this.itemHeight;

        if (itemTop < containerTop) {
          this.container.scrollTop = itemTop;
        } else if (itemBottom > containerBottom) {
          this.container.scrollTop =
            itemBottom - this.visibleCount * this.itemHeight;
        }
      }
    }
  }

    // ----------------- Initialize Cache -----------------
  const cache = new SuggestionCache();

    // ----------------- Quick Help Suggestions -----------------
    /**
     * Load quick help suggestions based on current language
     */ 
  loadQuickHelpSuggestions();

  // Main functions
async function loadQuickHelpSuggestions() {
  try {
    const cachedSuggestions = await cache.get('__quick_help__', currentLanguage);
    if (cachedSuggestions && cachedSuggestions.length > 0) {
      renderQuickHelp(cachedSuggestions.slice(0, 5));
      return;
    }

    const response = await fetch(`http://localhost:8000/suggestions?limit=5&lang=${currentLanguage}`);
    if (response.ok) {
      const data = await response.json();
      await cache.set('__quick_help__', data.suggestions, currentLanguage);
      renderQuickHelp(data.suggestions);
    } else {
      throw new Error('API request failed');
    }
  } catch (error) {
    console.error("Error loading suggestions:", error);
    const fallback = currentLanguage === 'DE' ? [
      "Was ist ARENA2036?",
      "Wie kann ich über das ARENA2036-Projekt lernen?",
      "Was bietet ARENA2036 für mich?",
      "Wie verwende ich ARENA2036 Projects?",
      "Wie passe ich mein ARENA2036-Profil an?"
    ] : [
      "What is ARENA2036 about?",
      "How can I learn about the ARENA2036 project?",
      "What does ARENA2036 offer for me?",
      "How do I use ARENA2036 Projects?",
      "How do I customize my ARENA2036 profile?"
    ];
    renderQuickHelp(fallback);
  }
}


function renderQuickHelp(suggestions) {
  // The loading element may have been re-created by switchLanguage(), so query it fresh
  const quickHelpLoadingEl = quickHelpContainer.querySelector("#quickHelpLoading");
  if (quickHelpLoadingEl) quickHelpLoadingEl.style.display = "none";

  // Build help tags safely (avoid injecting raw HTML directly if suggestions come from external source)
  quickHelpContainer.innerHTML = ""; // clear first
  suggestions.forEach((suggestion) => {
    const btn = document.createElement("button");
    btn.className = "help-tag";
    btn.dataset.question = suggestion;
    btn.textContent = suggestion;
    quickHelpContainer.appendChild(btn);
  });

  quickHelpContainer.querySelectorAll(".help-tag").forEach((tag) => {
    tag.addEventListener("click", () => {
      startChatFlow(tag.dataset.question);
    });
  });
}


    // ----------------- Autocomplete Functionality -----------------
    /**
     * Unified autocomplete handler for both search and chat inputs
     * @param {Object} params - Configuration object
     * @param {string} params.query - Search query
     * @param {string} params.inputType - Input type ('search' or 'chat')
     */
  async function loadUnifiedAutocomplete({
    query,
    inputType, // 'search' or 'chat'
  }) {
    if (query.length < 1) {
      if (inputType === "search") hideSuggestions();
      else hideChatSuggestions();
      return;
    }

    try {
      let suggestions = null;
      if (inputType === "search") {
        const cachedSuggestions = await cache.get(query, currentLanguage);
        if (cachedSuggestions) {
          suggestions = cachedSuggestions;
        } else {
          const response = await fetch(
            `http://localhost:8000/suggestions?q=${encodeURIComponent(
              query
            )}&limit=20&lang=${currentLanguage}`
          );
          if (response.ok) {
            const data = await response.json();
            await cache.set(query, data.suggestions, currentLanguage);
            suggestions = data.suggestions;
          }
        }
        showUnifiedAutocompleteSuggestions(suggestions || [], "search");
      } else {
        // chat input does not use cache
        const response = await fetch(
          `http://localhost:8000/suggestions?q=${encodeURIComponent(
            query
          )}&limit=20&lang=${currentLanguage}`
        );
        if (response.ok) {
          const data = await response.json();
          suggestions = data.suggestions;
        }
        showUnifiedAutocompleteSuggestions(suggestions || [], "chat");
      }
    } catch (error) {
      console.error("Error loading autocomplete suggestions:", error);
      if (inputType === "search") hideSuggestions();
      else hideChatSuggestions();
    }
  }

    /**
     * Show autocomplete suggestions
     * @param {Array} suggestions - List of suggestions
     * @param {string} inputType - Input type ('search' or 'chat')
     */
  function showUnifiedAutocompleteSuggestions(suggestions, inputType) {
    if (inputType === "search") {
      if (suggestions.length === 0) {
        hideSuggestions();
        return;
      }
      currentSuggestions = suggestions;
      selectedSuggestionIndex = -1;
      let suggestionsDropdown = document.getElementById("suggestionsDropdown");
      if (!suggestionsDropdown) {
        suggestionsDropdown = document.createElement("div");
        suggestionsDropdown.id = "suggestionsDropdown";
        suggestionsDropdown.className = "suggestions-dropdown";
        document
          .querySelector(".search-wrapper")
          .appendChild(suggestionsDropdown);
        virtualList = new VirtualSuggestionsList(suggestionsDropdown, searchInput);
        suggestionsDropdown.addEventListener("scroll", (e) => {
          virtualList.handleScroll(e.target.scrollTop);
        });
        suggestionsDropdown.addEventListener("click", handleSuggestionClick);
      }
      virtualList.render(suggestions);
      suggestionsDropdown.style.display = "block";
      suggestionsVisible = true;
    } else {
      if (suggestions.length === 0 || isAnswerLoading) {
        hideChatSuggestions();
        return;
      }
      currentChatSuggestions = suggestions;
      chatSelectedSuggestionIndex = -1;
      let chatSuggestionsDropdown = document.getElementById(
        "chatSuggestionsDropdown"
      );
      if (!chatSuggestionsDropdown) {
        chatSuggestionsDropdown = document.createElement("div");
        chatSuggestionsDropdown.id = "chatSuggestionsDropdown";
        chatSuggestionsDropdown.className = "chat-suggestions-dropdown";
        document
          .querySelector(".chat-input-container")
          .appendChild(chatSuggestionsDropdown);
        chatVirtualList = new VirtualSuggestionsList(chatSuggestionsDropdown, chatInput);
        chatSuggestionsDropdown.addEventListener("scroll", (e) => {
          chatVirtualList.handleScroll(e.target.scrollTop);
        });
        chatSuggestionsDropdown.addEventListener(
          "click",
          handleChatSuggestionClick
        );
      }
      chatVirtualList.render(suggestions);
      chatSuggestionsDropdown.style.display = "block";
      chatSuggestionsVisible = true;
    }
  }

    // ----------------- Suggestion Visibility Management -----------------
  function hideSuggestions() {
    const suggestionsDropdown = document.getElementById("suggestionsDropdown");
    if (suggestionsDropdown) {
      suggestionsDropdown.style.display = "none";
    }
    suggestionsVisible = false;
    selectedSuggestionIndex = -1;
  }

  function hideChatSuggestions() {
    const chatSuggestionsDropdown = document.getElementById(
      "chatSuggestionsDropdown"
    );
    if (chatSuggestionsDropdown) {
      chatSuggestionsDropdown.style.display = "none";
    }
    chatSuggestionsVisible = false;
    chatSelectedSuggestionIndex = -1;
  }

    // ----------------- Suggestion Interaction Handlers -----------------
  function handleSuggestionClick(e) {
    const item = e.target.closest(".suggestion-item");
    if (item) {
      const index = parseInt(item.dataset.index, 10);
      const suggestion = currentSuggestions[index];
      searchInput.value = suggestion;
      hideSuggestions();
      startChatFlow(suggestion);
    }
  }

  function handleChatSuggestionClick(e) {
    const item = e.target.closest(".suggestion-item");
    if (item) {
      const index = parseInt(item.dataset.index, 10);
      const suggestion = currentChatSuggestions[index];
      chatInput.value = suggestion;
      hideChatSuggestions();
      addChatMessage(suggestion);
      chatInput.value = "";
    }
  }

  function handleSuggestionKeyboard(e) {
    if (!suggestionsVisible || currentSuggestions.length === 0) return;

    if (e.key === "ArrowDown") {
      e.preventDefault();
      selectedSuggestionIndex = Math.min(
        selectedSuggestionIndex + 1,
        currentSuggestions.length - 1
      );
      virtualList.updateSelection(selectedSuggestionIndex);
      searchInput.value = currentSuggestions[selectedSuggestionIndex];
    } else if (e.key === "ArrowUp") {
      e.preventDefault();
      selectedSuggestionIndex = Math.max(selectedSuggestionIndex - 1, -1);
      if (selectedSuggestionIndex >= 0) {
        virtualList.updateSelection(selectedSuggestionIndex);
        searchInput.value = currentSuggestions[selectedSuggestionIndex];
      } else {
        virtualList.updateSelection(-1);
        searchInput.value = searchInput.dataset.originalValue || "";
      }
    } else if (e.key === "Enter" && selectedSuggestionIndex >= 0) {
      e.preventDefault();
      const selectedSuggestion = currentSuggestions[selectedSuggestionIndex];
      searchInput.value = selectedSuggestion;
      hideSuggestions();
      startChatFlow(selectedSuggestion);
    } else if (e.key === "Escape") {
      hideSuggestions();
      searchInput.value = searchInput.dataset.originalValue || "";
    }
  }

  function handleChatSuggestionKeyboard(e) {
    if (!chatSuggestionsVisible || currentChatSuggestions.length === 0) return;

    if (e.key === "ArrowDown") {
      e.preventDefault();
      chatSelectedSuggestionIndex = Math.min(
        chatSelectedSuggestionIndex + 1,
        currentChatSuggestions.length - 1
      );
      chatVirtualList.updateSelection(chatSelectedSuggestionIndex);
      chatInput.value = currentChatSuggestions[chatSelectedSuggestionIndex];
    } else if (e.key === "ArrowUp") {
      e.preventDefault();
      chatSelectedSuggestionIndex = Math.max(
        chatSelectedSuggestionIndex - 1,
        -1
      );
      if (chatSelectedSuggestionIndex >= 0) {
        chatVirtualList.updateSelection(chatSelectedSuggestionIndex);
        chatInput.value = currentChatSuggestions[chatSelectedSuggestionIndex];
      } else {
        chatVirtualList.updateSelection(-1);
        chatInput.value = chatInput.dataset.originalValue || "";
      }
    } else if (e.key === "Enter" && chatSelectedSuggestionIndex >= 0) {
      e.preventDefault();
      const selectedSuggestion =
        currentChatSuggestions[chatSelectedSuggestionIndex];
      chatInput.value = selectedSuggestion;
      hideChatSuggestions();
      addChatMessage(selectedSuggestion);
      chatInput.value = "";
    } else if (e.key === "Escape") {
      hideChatSuggestions();
      chatInput.value = chatInput.dataset.originalValue || "";
    }
  }

    // ----------------- Chat Flow Management -----------------
    /**
     * Start chat flow with a question
     * @param {string} question - User's question
     */
  function startChatFlow(question) {
    // Hide hero section and show chat flow
    heroSection.style.display = "none";
    chatFlow.style.display = "block";

    // Create and add new message
    addChatMessage(question);

    // Focus chat input and scroll to bottom
    setTimeout(() => {
      chatInput.focus();
      scrollToBottom();
    }, 100);
  }
  function addChatMessage(question) {
    if (isAnswerLoading) return;

    messageCounter++;
    const messageId = `message-${messageCounter}`;

    isAnswerLoading = true;
    updateInputStates();

    const messageElement = document.createElement("div");
    messageElement.className = "chat-message";
    messageElement.id = messageId;

    // Remove the isImageRequest check - show stop button for all responses
    const headerIcon = "✨";
    const headerLabel = currentLanguage === 'DE' ? 'Antwort' : 'Answer';
    const loadingMessage = currentLanguage === 'DE' 
        ? 'Ich suche die beste Antwort für Sie...' 
        : 'Finding the best answer for you...';

    messageElement.innerHTML = `
        <div class="question-display">${question}</div>
        <div class="answer-section">
            <div class="answer-header">
                <span class="answer-icon">${headerIcon}</span>
                <span class="answer-label">${headerLabel}</span>
                <button aria-label="${currentLanguage === 'DE' ? 'Generierung stoppen' : 'Stop generation'}" class="stop-generation-btn" onclick="stopGeneration('${messageId}')">
                    <svg viewBox="0 0 24 24" width="16" height="16">
                        <rect x="8" y="8" width="8" height="8" fill="currentColor" rx="2"/>
                    </svg>
                </button>
            </div>
            <div class="loading-state">
                <div class="loading-spinner"></div>
                <p class="loading-text">${loadingMessage}</p>
            </div>
        </div>
    `;

    conversationHistory.appendChild(messageElement);
    fetchAnswerForMessage(question, messageId);
    setTimeout(() => {
        scrollToBottom();
    }, 100);
}

    /**
     * Fetch answer for a chat message
     * @param {string} question - User's question
     * @param {string} messageId - DOM element ID of the message
     */
  const fetchAnswerForMessage = async (question, messageId) => {
    const messageElement = document.getElementById(messageId);
    if (!messageElement) {
    console.warn('Message element removed before response', messageId);
    return;
    }
    const answerSection = messageElement.querySelector(".answer-section");


    currentAbortController = new AbortController();

    try {
      const response = await fetch(
        `http://localhost:8000/query?question=${encodeURIComponent(
          question
        )}&lang=${currentLanguage}`,
        {
          signal: currentAbortController.signal,
        }
      );

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();

      // Check if response is an image
      renderAnswerInMessage(
        answerSection,
        data.answer,
        data.sources,
        question,
        data.is_image
      );
    } catch (error) {
      if (error.name === "AbortError") {
        const abortMd = currentLanguage === 'DE'
          ? "## Generierung gestoppt\n\nDie Generierung wurde vom Benutzer abgebrochen."
          : "## Answer Stopped\n\nGeneration was stopped by user.";

        renderAnswerInMessage(
          answerSection,
          abortMd,
          [],
          question,
          false
        );
      } else {
        console.error("Error:", error);
        let errorMessage = currentLanguage === 'DE'
          ? "Ich habe gerade Schwierigkeiten, eine Verbindung herzustellen. Bitte versuchen Sie es in einem Moment erneut."
          : "I'm having trouble connecting right now. Please try again in a moment.";
        if (error.message.includes("Failed to fetch")) {
           errorMessage = currentLanguage === 'DE'
            ? "Verbindung zum Server fehlgeschlagen. Bitte prüfen Sie Ihre Verbindung und versuchen Sie es erneut."
            : "Unable to connect to the server. Please check your connection and try again.";
        } else if (error.message.includes("500")) {
           errorMessage = currentLanguage === 'DE'
           ? "Der Server hat ein internes Problem festgestellt. Bitte versuchen Sie es später erneut."
           : "The server encountered an internal error. Please try again later.";

        }
        renderAnswerInMessage(
          answerSection,
          `## Error\n\n${errorMessage}`,
          [],
          question,
          false
        );
      }
    } finally {
      isAnswerLoading = false;
      currentAbortController = null;
      updateInputStates();
    }
  };

    /**
     * Render answer in message container
     * @param {HTMLElement} answerSection - DOM element to render into
     * @param {string} markdownText - Answer in markdown format
     * @param {Array} sources - List of source objects
     * @param {string} question - Original question
     * @param {boolean} isImage - Whether answer is an image
     */
function renderAnswerInMessage(answerSection, markdownText, sources, question, isImage = false) {
    let htmlContent;
    
    if (isImage) {
    // Safe regex matches
    const imgMatch = markdownText && markdownText.match(/!\[.*\]\((.*?)\)/);
    const altMatch = markdownText && markdownText.match(/!\[(.*?)\]/);

    const imgUrl = imgMatch ? imgMatch[1] : null;
    const altText = altMatch ? altMatch[1] : (currentLanguage === 'DE' ? 'Generiertes Bild' : 'Generated Image');
    const imageHeader = currentLanguage === 'DE' ? 'Generiertes Bild' : 'Generated Image';

    if (!imgUrl) {
        // No URL found — show friendly localized message
        answerSection.innerHTML = `<div class="answer-content"><p>${currentLanguage === 'DE' ? 'Kein Bild gefunden.' : 'No image URL found.'}</p></div>`;
        scrollToBottom();
        return;
    }

    // Fetch image as blob and display (handle errors)
    fetch(`http://localhost:8000/serve-image?path=${encodeURIComponent(imgUrl)}`)
        .then(response => {
        if (!response.ok) throw new Error('Image fetch failed');
        return response.blob();
        })
        .then(blob => {
        const reader = new FileReader();
        reader.onloadend = () => {
            answerSection.innerHTML = `
            <div class="answer-header">
                <span class="answer-icon">🖼️</span>
                <span class="answer-label">${imageHeader}</span>
            </div>
            <div class="answer-content">
                <img src="${reader.result}" alt="${altText}" style="max-width: 100%; border-radius: 8px; margin-bottom: 12px;">
                <p>${altText}</p>
            </div>
            `;
            scrollToBottom();
        };
        reader.readAsDataURL(blob);
        })
        .catch(error => {
        console.error("Image fetch error:", error);
        answerSection.innerHTML = `<div class="answer-content"><p>${currentLanguage === 'DE' ? 'Fehler beim Laden des Bildes.' : 'Error loading image.'}</p></div>`;
        scrollToBottom();
        });

    // Important: return so the later `answerSection.innerHTML = htmlContent;` does not overwrite image content
    return;
    } else {
        const parsedHtml = marked.parse(markdownText);
        // Language-specific UI elements
        const sourcesTitle = currentLanguage === 'DE' ? 'Quellen' : 'Sources';
        const feedbackText = currentLanguage === 'DE' ? 'War diese Antwort hilfreich?' : 'Was this answer helpful?';
        
        htmlContent = `
            <div class="answer-header">
                <span class="answer-icon">✨</span>
                <span class="answer-label">${currentLanguage === 'DE' ? 'Antwort' : 'Answer'}</span>
            </div>
            <div class="answer-content">${parsedHtml}</div>
            ${sources && sources.length > 0 ? `
                <div class="sources-section">
                    <h4 class="sources-title">${sourcesTitle}</h4>
                    <div class="sources-list">
                        ${sources.map(source => `
                            <a href="${source.url}" target="_blank" rel="noopener noreferrer" class="source-item">
                                <span class="source-icon">🔗</span>
                                <span class="source-title">${source.title || (currentLanguage === 'DE' ? 'Quelle' : 'Source')}</span>
                            </a>
                        `).join('')}
                    </div>
                </div>
            ` : ''}
            <div class="answer-feedback">
                <span class="feedback-text">${feedbackText}</span>
                <div class="feedback-buttons">
                    <button class="feedback-btn feedback-yes" aria-label="${currentLanguage === 'DE' ? 'Hilfreich' : 'Helpful'}">👍</button>
                    <button class="feedback-btn feedback-no" aria-label="${currentLanguage === 'DE' ? 'Nicht hilfreich' : 'Not helpful'}">👎</button>

                </div>
            </div>
        `;
    }
    
    if (!isImage) {
    answerSection.innerHTML = htmlContent;
    }
    
    // Highlight code blocks for non-image responses
    if (!isImage) {
        answerSection.querySelectorAll('pre code').forEach((block) => {
            hljs.highlightElement(block);
        });
    }
    
    // Language-specific loading message for related questions
    const loadingText = currentLanguage === 'DE' 
        ? 'Lade verwandte Fragen...' 
        : 'Loading related questions...';
    relatedQuestionsSection.style.display = "block";
    relatedQuestionsList.innerHTML = `
        <div class="loading-state">
            <div class="loading-spinner"></div>
            <p class="loading-text">${loadingText}</p>
        </div>
    `;
    loadRelatedQuestions(question);
    
    scrollToBottom();
}

    // ----------------- Related Questions -----------------
    /**
     * Load questions related to the current one
     * @param {string} currentQuestion - The current question
     */
async function loadRelatedQuestions(currentQuestion) {
    try {
        const response = await fetch(
            `http://localhost:8000/related-questions?question=${encodeURIComponent(currentQuestion)}&lang=${currentLanguage}`
        );
        
        if (response.ok) {
            const data = await response.json();
            renderRelatedQuestions(data.related_questions);
        } else {
            // Language-specific fallback questions
            const fallback = currentLanguage === 'DE' ? [
                "Wie verwalte ich ARENA2036-Benachrichtigungen?",
                "Was sind die ARENA2036-Kollaborationsfunktionen?",
                "Wie integriere ich Drittanbietertools mit ARENA2036?",
                "Wie exportiere ich Daten aus ARENA2036?"
            ] : [
                "How do I manage Arena2036 notifications?",
                "What are the Arena2036 collaboration features?",
                "How do I integrate third-party tools with Arena2036?",
                "How do I export data from Arena2036?"
            ];
            renderRelatedQuestions(fallback);
        }
    } catch (error) {
        console.error("Error loading related questions:", error);
        // Same fallback as above
        const fallback = currentLanguage === 'DE' ? [
            "Wie verwalte ich ARENA2036-Benachrichtigungen?",
            "Was sind die ARENA2036-Kollaborationsfunktionen?",
            "Wie integriere ich Drittanbietertools mit ARENA2036?",
            "Wie exportiere ich Daten aus ARENA2036?"
        ] : [
            "How do I manage Arena2036 notifications?",
            "What are the Arena2036 collaboration features?",
            "How do I integrate third-party tools with Arena2036?",
            "How do I export data from Arena2036?"
        ];
        renderRelatedQuestions(fallback);
    }
}

function renderRelatedQuestions(questions) {
  relatedQuestionsList.innerHTML = ""; // Clear previous ones
  questions.forEach((q) => {
    const btn = document.createElement('button');
    btn.className = 'related-question';
    btn.dataset.question = q;
    btn.textContent = q;
    relatedQuestionsList.appendChild(btn);
  });

  relatedQuestionsSection.style.display = "block";

  relatedQuestionsList
    .querySelectorAll(".related-question")
    .forEach((questionBtn) => {
      questionBtn.addEventListener("click", () => {
        addChatMessage(questionBtn.dataset.question);
        chatInput.value = "";
      });
    });
}


    // ----------------- Utility Functions -----------------
  function scrollToBottom() {
    setTimeout(() => {
      window.scrollTo({
        top: document.body.scrollHeight,
        behavior: "smooth",
      });
    }, 50);
  }

  function stopGeneration(messageId) {
    if (currentAbortController) {
      currentAbortController.abort();
    }

    // Hide stop button
    const stopBtn = document.querySelector(
      `#${messageId} .stop-generation-btn`
    );
    if (stopBtn) {
      stopBtn.style.display = "none";
    }
  }

function updateInputStates() {
  const searchInputContainer = document.querySelector('.search-wrapper');
  const chatInputContainer = document.querySelector('.chat-input-container');

  if (isAnswerLoading) {
    const waitText = currentLanguage === 'DE'
      ? "Bitte warten Sie auf die aktuelle Antwort..."
      : "Please wait for the current answer to complete...";

    searchInput.placeholder = waitText;
    chatInput.placeholder = waitText;

    searchInput.disabled = true;
    chatInput.disabled = true;
    searchButton.disabled = true;
    chatSendButton.disabled = true;

    searchInputContainer.classList.add('disabled');
    chatInputContainer.classList.add('disabled');

    hideChatSuggestions();
  } else {
    // Restore placeholders from data attributes if present
    searchInput.placeholder = searchInput.getAttribute(`data-placeholder-${currentLanguage.toLowerCase()}`) || '';
    chatInput.placeholder = currentLanguage === 'DE' ? "Stellen Sie eine Folgefrage..." : "Ask a follow-up question...";

    searchInput.disabled = false;
    chatInput.disabled = false;
    searchButton.disabled = false;
    chatSendButton.disabled = false;

    searchInputContainer.classList.remove('disabled');
    chatInputContainer.classList.remove('disabled');
  }
}

    // ----------------- Global Functions -----------------
  window.handleFeedback = function (button, isPositive) {
    const feedbackButtons = button.parentNode;
    feedbackButtons
      .querySelectorAll(".feedback-btn")
      .forEach((btn) => btn.classList.remove("active"));
    button.classList.add("active");
  };

  // Make stopGeneration function global
  window.stopGeneration = stopGeneration;

    // ----------------- Event Listeners -----------------
  searchInput.addEventListener("input", (e) => {
    const query = e.target.value.trim();
    searchInput.dataset.originalValue = query;
    if (suggestionTimeout) {
      clearTimeout(suggestionTimeout);
    }
    suggestionTimeout = setTimeout(() => {
      loadUnifiedAutocomplete({ query, inputType: "search" });
    }, 150);
  });

  searchInput.addEventListener("keydown", handleSuggestionKeyboard);

  searchInput.addEventListener("focus", () => {
    const query = searchInput.value.trim();
    if (query.length >= 1) {
      loadUnifiedAutocomplete({ query, inputType: "search" });
    }
  });

  document.addEventListener("click", (e) => {
    if (!e.target.closest(".search-wrapper")) {
      hideSuggestions();
    }
    if (!e.target.closest(".chat-input-container")) {
      hideChatSuggestions();
    }
  });

  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape") {
      if (isAnswerLoading && currentAbortController) {
        stopGeneration(`message-${messageCounter}`);
      }
      hideSuggestions();
      hideChatSuggestions();
    }
  });

  searchButton.addEventListener("click", () => {
    const question = searchInput.value.trim();
    if (question) {
      hideSuggestions();
      startChatFlow(question);
    }
  });

  searchInput.addEventListener("keypress", (e) => {
    if (e.key === "Enter" && !suggestionsVisible) {
      const question = searchInput.value.trim();
      if (question) {
        hideSuggestions();
        startChatFlow(question);
      }
    }
  });

  chatInput.addEventListener("input", (e) => {
    if (isAnswerLoading) return;
    const query = e.target.value.trim();
    chatInput.dataset.originalValue = query;
    if (suggestionTimeout) {
      clearTimeout(suggestionTimeout);
    }
    suggestionTimeout = setTimeout(() => {
      loadUnifiedAutocomplete({ query, inputType: "chat" });
    }, 150);
  });

  chatInput.addEventListener("keydown", (e) => {
    if (isAnswerLoading && e.key !== "Escape") {
      e.preventDefault();
      return;
    }
    handleChatSuggestionKeyboard(e);
  });

  chatInput.addEventListener("focus", () => {
    if (isAnswerLoading) return;
    const query = chatInput.value.trim();
    if (query.length >= 1) {
      loadUnifiedAutocomplete({ query, inputType: "chat" });
    }
  });

  chatSendButton.addEventListener("click", () => {
    if (isAnswerLoading) return;

    const question = chatInput.value.trim();
    if (question) {
      hideChatSuggestions();
      addChatMessage(question);
      chatInput.value = "";
    }
  });

  chatInput.addEventListener("keypress", (e) => {
    if (e.key === "Enter" && !chatSuggestionsVisible && !isAnswerLoading) {
      const question = chatInput.value.trim();
      if (question) {
        hideChatSuggestions();
        addChatMessage(question);
        chatInput.value = "";
      }
    }
  });
    // ----------------- Language Switching -----------------
  const btnEn = document.getElementById("lang-en");
  const btnDe = document.getElementById("lang-de");
  btnEn.addEventListener("click", () => {
    btnEn.classList.add("active");
    btnDe.classList.remove("active");
    switchLanguage("EN");
  });
  btnDe.addEventListener("click", () => {
    btnDe.classList.add("active");
    btnEn.classList.remove("active");
    switchLanguage("DE");
  });

    // ----------------- Initialization -----------------
    // Focus search input on page load
  searchInput.focus();
});
