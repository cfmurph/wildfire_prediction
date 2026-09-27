import type { Config } from "tailwindcss";

const config: Config = {
  content: [
    "./app/**/*.{js,ts,jsx,tsx,mdx}",
    "./components/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: {
    extend: {
      colors: {
        fire: {
          low:      "#22c55e",
          moderate: "#eab308",
          high:     "#f97316",
          vhigh:    "#ef4444",
          extreme:  "#7c3aed",
        },
      },
    },
  },
  plugins: [],
};

export default config;
