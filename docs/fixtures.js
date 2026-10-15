window.POS_DEMO_FIXTURES = [
  {
    sentence: "I can fish.",
    focusToken: 1,
    tokens: [
      { text: "I", tag: "PRON" },
      { text: "can", tag: "AUX" },
      { text: "fish", tag: "VERB" },
      { text: ".", tag: "PUNCT" }
    ]
  },
  {
    sentence: "A can of fish.",
    focusToken: 1,
    tokens: [
      { text: "A", tag: "DET" },
      { text: "can", tag: "NOUN" },
      { text: "of", tag: "ADP" },
      { text: "fish", tag: "NOUN" },
      { text: ".", tag: "PUNCT" }
    ]
  }
];

window.POS_TAGS = {
  ADJ: { name: "Adjective", color: "pink" },
  ADP: { name: "Adposition", color: "blue" },
  ADV: { name: "Adverb", color: "teal" },
  AUX: { name: "Auxiliary", color: "violet" },
  CCONJ: { name: "Coordinating conjunction", color: "gold" },
  DET: { name: "Determiner", color: "amber" },
  INTJ: { name: "Interjection", color: "rose" },
  NOUN: { name: "Noun", color: "coral" },
  NUM: { name: "Numeral", color: "indigo" },
  PART: { name: "Particle", color: "lime" },
  PRON: { name: "Pronoun", color: "cyan" },
  PROPN: { name: "Proper noun", color: "orange" },
  PUNCT: { name: "Punctuation", color: "slate" },
  SCONJ: { name: "Subordinating conjunction", color: "gold" },
  SYM: { name: "Symbol", color: "slate" },
  VERB: { name: "Verb", color: "green" },
  X: { name: "Other", color: "slate" }
};
