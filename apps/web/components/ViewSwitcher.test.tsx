import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import ViewSwitcher from "./ViewSwitcher";

describe("ViewSwitcher", () => {
  it("switches views and shows the history year slider only then", () => {
    const onChange = vi.fn();
    const onYearChange = vi.fn();
    const { rerender } = render(
      <ViewSwitcher current="current" onChange={onChange} historyYear={2023} onYearChange={onYearChange} />,
    );
    expect(screen.getByRole("button", { name: /Current/ })).toBeEnabled();
    expect(screen.queryByRole("slider")).not.toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: /History/ }));
    expect(onChange).toHaveBeenCalledWith("history");

    rerender(
      <ViewSwitcher current="history" onChange={onChange} historyYear={2023} onYearChange={onYearChange} />,
    );
    const slider = screen.getByRole("slider");
    expect(slider).toHaveValue("2023");
    fireEvent.change(slider, { target: { value: "2017" } });
    expect(onYearChange).toHaveBeenCalledWith(2017);

    fireEvent.click(screen.getByRole("button", { name: /Predictions/ }));
    fireEvent.click(screen.getByRole("button", { name: /Risk/ }));
    expect(onChange).toHaveBeenCalledWith("predictions");
    expect(onChange).toHaveBeenCalledWith("risk");
  });
});
