import { NavLink, Outlet } from "react-router-dom";

export function AppShell() {
  return (
    <div className="app-shell">
      <nav className="app-shell__nav">
        <span className="app-shell__brand">tracee</span>
        <div className="app-shell__links">
          <NavLink
            className={({ isActive }) => `app-shell__link${isActive ? " is-active" : ""}`}
            end
            to="."
          >
            Graph
          </NavLink>
          <NavLink
            className={({ isActive }) => `app-shell__link${isActive ? " is-active" : ""}`}
            to="playground"
          >
            Playground
          </NavLink>
          <NavLink
            className={({ isActive }) => `app-shell__link${isActive ? " is-active" : ""}`}
            to="prompts"
          >
            Prompts
          </NavLink>
          <NavLink
            className={({ isActive }) => `app-shell__link${isActive ? " is-active" : ""}`}
            to="docs"
          >
            Docs
          </NavLink>
        </div>
      </nav>
      <main className="app-shell__content">
        <Outlet />
      </main>
    </div>
  );
}
