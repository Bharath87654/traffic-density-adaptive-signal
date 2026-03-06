/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}", // <-- If this is missing, no styles will load
  ],
  theme: {
    extend: {},
  },
  plugins: [],
}