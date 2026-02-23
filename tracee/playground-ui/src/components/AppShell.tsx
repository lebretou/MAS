import { NavLink, Outlet } from "react-router-dom";

export function AppShell() {
  return (
    <div className="app-shell">
      <nav className="app-shell__nav">
        <span className="app-shell__brand">tracee</span>
        <div className="app-shell__links">
          <NavLink to="/" end className={({ isActive }) => `app-shell__link ${isActive ? "is-active" : ""}`}>
            Graph
          </NavLink>
          <NavLink to="/playground" className={({ isActive }) => `app-shell__link ${isActive ? "is-active" : ""}`}>
            Playground
          </NavLink>
        </div>
      </nav>
      <main className="app-shell__content">
        <Outlet />
      </main>
    </div>
  );
}
