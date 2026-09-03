import App from "./App.svelte";
import "./app.css";
import "./workspace.css";
import { mount } from "svelte";
mount(App, { target: document.getElementById("app")! });
