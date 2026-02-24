import { Outlet } from "react-router-dom";

export function AppShell() {
  return (
    <div className="app-shell">
      <main className="app-shell__content">
        <Outlet />
      </main>
    </div>
  );
}
