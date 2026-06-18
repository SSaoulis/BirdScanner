import type { Config } from "tailwindcss";

/**
 * BirdScanner "field journal" theme.
 *
 * The palette is grounded in a naturalist's printed bird guide: aged paper,
 * forest ink, foliage sage, and a single goldfinch-ochre "spotting" accent.
 * Rust is reserved for destructive actions. Type pairs Fraunces (a soft
 * literary serif, used for species names + headings) with Hanken Grotesk
 * (a friendly humanist sans for body + controls + data).
 */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        paper: "#E9E2CF", // app background — green-tinted aged paper
        card: "#F6F1E2", // lifted card stock, a shade lighter than the page
        line: "#D8CDB0", // tea-stain hairlines, rules, borders
        ink: "#2C3A2E", // primary text — deep forest ink
        bark: "#6E6448", // secondary text — warm brown
        sage: "#82906F", // muted foliage green — structure / quiet accents
        "sage-deep": "#566048", // sage dark enough for text on paper
        gold: "#C08A2D", // goldfinch ochre — the spotting accent (fills)
        "gold-deep": "#8A6113", // gold dark enough for accent text on paper
        rust: "#A4402A", // destructive actions only
      },
      fontFamily: {
        display: ['"Fraunces"', "Georgia", "serif"],
        sans: ['"Hanken Grotesk"', "system-ui", "sans-serif"],
      },
      boxShadow: {
        // Soft, warm-tinted lift — like a photo resting on a journal page.
        plate: "0 1px 2px rgba(44,58,46,0.06), 0 6px 16px rgba(44,58,46,0.10)",
        "plate-lift": "0 2px 4px rgba(44,58,46,0.08), 0 14px 30px rgba(44,58,46,0.16)",
      },
    },
  },
  plugins: [],
} satisfies Config;
