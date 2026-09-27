import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { viewById } from "@/lib/views";
import ComingSoon from "./ComingSoon";

describe("ComingSoon", () => {
  it("tells the reader the view is not in this release", () => {
    const { rerender } = render(<ComingSoon view={viewById("history")} />);
    expect(screen.getByRole("status")).toHaveTextContent(/not in the current release/i);
    expect(screen.getAllByText(/time slider/i).length).toBeGreaterThan(0);
    expect(screen.getByRole("heading", { name: "History" })).toBeInTheDocument();

    rerender(<ComingSoon view={viewById("risk")} />);
    expect(screen.getByRole("heading", { name: "Long-term risk" })).toBeInTheDocument();
    expect(screen.queryAllByText(/time slider/i)).toHaveLength(0);

    rerender(<ComingSoon view={viewById("predict")} />);
    expect(screen.getByRole("heading", { name: "Short-term" })).toBeInTheDocument();
    expect(screen.getByText(/not in the current release/i)).toBeInTheDocument();
  });
});
