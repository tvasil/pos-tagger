(function () {
  "use strict";

  const fixtures = window.POS_DEMO_FIXTURES;
  const tagDefinitions = window.POS_TAGS;

  const form = document.querySelector("#sentence-form");
  const input = document.querySelector("#sentence-input");
  const status = document.querySelector("#form-status");
  const tokenList = document.querySelector("#token-list");
  const tokenCount = document.querySelector("#token-count");
  const legend = document.querySelector("#tag-legend");
  const inspectorTitle = document.querySelector("#inspector-title");
  const inspectorTag = document.querySelector("#inspector-tag");
  const inspectorSummary = document.querySelector("#inspector-summary");
  const featureGroups = document.querySelector("#feature-groups");
  const comparisonCards = document.querySelector("#comparison-cards");

  let currentFixture = fixtures[0];
  let selectedIndex = currentFixture.focusToken;

  function normalizeSentence(value) {
    return value.trim().replace(/\s+/g, " ").replace(/\s+([.!?,;:])/g, "$1").toLowerCase();
  }

  function findFixture(value) {
    const normalized = normalizeSentence(value);
    return fixtures.find((fixture) => normalizeSentence(fixture.sentence) === normalized);
  }

  function isUpper(value) {
    return /\p{L}/u.test(value) && value === value.toLocaleUpperCase();
  }

  function isLower(value) {
    return /\p{L}/u.test(value) && value === value.toLocaleLowerCase();
  }

  function isTitle(value) {
    if (!value || !/\p{L}/u.test(value[0])) return false;
    return value[0] === value[0].toLocaleUpperCase() && value.slice(1) === value.slice(1).toLocaleLowerCase();
  }

  function getStructure(value) {
    return Array.from(value, (character) => {
      if (/\p{N}/u.test(character)) return "d";
      if (/\p{Lu}/u.test(character)) return "X";
      if (/\p{Ll}/u.test(character)) return "x";
      return character;
    }).join("");
  }

  function extractFeatures(tokens, index) {
    const word = tokens[index].text;
    const features = {
      lexical: [
        ["Original", word],
        ["Lowercase", word.toLocaleLowerCase()],
        ["Prefix · 1", word.slice(0, 1)],
        ["Prefix · 2", word.slice(0, 2)],
        ["Prefix · 3", word.slice(0, 3)],
        ["Suffix · 1", word.slice(-1)],
        ["Suffix · 2", word.slice(-2)],
        ["Suffix · 3", word.slice(-3)]
      ],
      shape: [
        ["Structure", getStructure(word)],
        ["Title case", isTitle(word)],
        ["All caps", isUpper(word)],
        ["All lowercase", isLower(word)],
        ["Alphabetic", /^\p{L}+$/u.test(word)],
        ["Alphanumeric", /^[\p{L}\p{N}]+$/u.test(word)],
        ["Numeric", /^\p{N}+$/u.test(word)],
        ["Has hyphen", word.includes("-")],
        ["Capitals inside", word.slice(1).toLocaleLowerCase() !== word.slice(1)]
      ],
      position: [
        ["First token", index === 0],
        ["Last token", index === tokens.length - 1]
      ],
      context: []
    };

    if (index > 0) {
      const previous = tokens[index - 1].text;
      features.context.push(
        ["Previous word", previous],
        ["Previous lowercase", previous.toLocaleLowerCase()],
        ["Previous is title case", isTitle(previous)],
        ["Previous is first", index - 1 === 0]
      );
    }

    if (index < tokens.length - 1) {
      const next = tokens[index + 1].text;
      features.context.push(
        ["Next word", next],
        ["Next lowercase", next.toLocaleLowerCase()],
        ["Next is title case", isTitle(next)],
        ["Next is last", index + 1 === tokens.length - 1]
      );
    }

    return features;
  }

  function createTokenButton(token, index) {
    const definition = tagDefinitions[token.tag];
    const button = document.createElement("button");
    button.type = "button";
    button.className = `token token--${definition.color}`;
    button.dataset.index = String(index);
    button.setAttribute("aria-pressed", String(index === selectedIndex));
    button.setAttribute("aria-label", `${token.text}, ${definition.name}. Inspect features.`);
    button.innerHTML = `<span class="token-word"></span><span class="token-tag"></span>`;
    button.querySelector(".token-word").textContent = token.text;
    button.querySelector(".token-tag").textContent = token.tag;
    button.addEventListener("click", () => selectToken(index));
    return button;
  }

  function renderTokens() {
    tokenList.replaceChildren();
    currentFixture.tokens.forEach((token, index) => tokenList.append(createTokenButton(token, index)));
    tokenCount.textContent = `${currentFixture.tokens.length} tokens`;
  }

  function renderLegend() {
    const usedTags = [...new Set(currentFixture.tokens.map((token) => token.tag))];
    legend.replaceChildren();

    usedTags.forEach((tag) => {
      const definition = tagDefinitions[tag];
      const item = document.createElement("li");
      item.innerHTML = `<span class="legend-dot legend-dot--${definition.color}"></span><strong></strong><span></span>`;
      item.querySelector("strong").textContent = tag;
      item.querySelector("span:last-child").textContent = definition.name;
      legend.append(item);
    });
  }

  function formatFeatureValue(value) {
    if (typeof value !== "boolean") return value;
    return value ? "Yes" : "No";
  }

  function renderFeatureGroup(title, rows) {
    if (!rows.length) return null;
    const section = document.createElement("section");
    section.className = "feature-group";
    const heading = document.createElement("h4");
    heading.textContent = title;
    const list = document.createElement("dl");

    rows.forEach(([label, value]) => {
      const row = document.createElement("div");
      const term = document.createElement("dt");
      const description = document.createElement("dd");
      term.textContent = label;
      description.textContent = formatFeatureValue(value);
      if (typeof value === "boolean") description.className = value ? "value-yes" : "value-no";
      row.append(term, description);
      list.append(row);
    });

    section.append(heading, list);
    return section;
  }

  function renderInspector() {
    const token = currentFixture.tokens[selectedIndex];
    const definition = tagDefinitions[token.tag];
    const features = extractFeatures(currentFixture.tokens, selectedIndex);

    inspectorTitle.textContent = `“${token.text}”`;
    inspectorTag.textContent = `${token.tag} · ${definition.name}`;
    inspectorTag.className = `inspector-tag inspector-tag--${definition.color}`;
    inspectorSummary.textContent = `These observations are available to the CRF when it scores the tag sequence. They are not independent rules.`;
    featureGroups.replaceChildren();

    [
      ["Word identity", features.lexical],
      ["Word structure", features.shape],
      ["Sentence position", features.position],
      ["Neighboring words", features.context]
    ].forEach(([title, rows]) => featureGroups.append(renderFeatureGroup(title, rows)));
  }

  function selectToken(index) {
    selectedIndex = index;
    tokenList.querySelectorAll(".token").forEach((button) => {
      button.setAttribute("aria-pressed", String(Number(button.dataset.index) === selectedIndex));
    });
    renderInspector();
  }

  function renderFixture(fixture) {
    currentFixture = fixture;
    selectedIndex = fixture.focusToken;
    input.value = fixture.sentence;
    status.textContent = "";
    renderTokens();
    renderLegend();
    renderInspector();
  }

  function renderComparison() {
    comparisonCards.replaceChildren();
    fixtures.forEach((fixture) => {
      const token = fixture.tokens[fixture.focusToken];
      const definition = tagDefinitions[token.tag];
      const features = extractFeatures(fixture.tokens, fixture.focusToken);
      const previous = features.context.find(([label]) => label === "Previous word")[1];
      const next = features.context.find(([label]) => label === "Next word")[1];
      const card = document.createElement("article");
      card.className = "comparison-card";
      card.innerHTML = `
        <p class="comparison-sentence"></p>
        <div class="comparison-result">
          <span class="comparison-word"></span>
          <span class="comparison-arrow" aria-hidden="true">→</span>
          <span class="comparison-tag"></span>
        </div>
        <dl>
          <div><dt>Previous word</dt><dd class="previous-word"></dd></div>
          <div><dt>Next word</dt><dd class="next-word"></dd></div>
        </dl>
      `;
      card.querySelector(".comparison-sentence").textContent = fixture.sentence;
      card.querySelector(".comparison-word").textContent = token.text;
      card.querySelector(".comparison-tag").textContent = `${token.tag} · ${definition.name}`;
      card.querySelector(".comparison-tag").classList.add(`comparison-tag--${definition.color}`);
      card.querySelector(".previous-word").textContent = previous;
      card.querySelector(".next-word").textContent = next;
      comparisonCards.append(card);
    });
  }

  form.addEventListener("submit", (event) => {
    event.preventDefault();
    const fixture = findFixture(input.value);
    if (fixture) {
      renderFixture(fixture);
      return;
    }

    status.textContent = "This prototype only has predictions for the two examples. Choose one to continue.";
    status.focus();
  });

  document.querySelectorAll(".example-button").forEach((button) => {
    button.addEventListener("click", () => renderFixture(findFixture(button.dataset.example)));
  });

  renderFixture(currentFixture);
  renderComparison();
})();
