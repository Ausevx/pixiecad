import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { App } from "./App";
import { initTheme } from "./lib/theme";
import "./styles.css";

// Before render: a theme applied in an effect would flash the wrong one.
initTheme();

const container = document.getElementById("root");
if (!container) throw new Error("#root is missing from the document");

createRoot(container).render(
  <StrictMode>
    <App />
  </StrictMode>,
);
