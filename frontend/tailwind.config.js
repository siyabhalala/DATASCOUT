/** @type {import('tailwindcss').Config} */
module.exports = {
  content: ['./src/**/*.{js,ts,jsx,tsx}'],
  theme: {
    extend: {
      fontFamily: { sans: ['Inter', 'system-ui', 'sans-serif'] },
      colors: {
        brand: { 50:'#f0f7ff', 100:'#e0effe', 500:'#3b82f6', 600:'#2563eb', 700:'#1d4ed8' }
      }
    }
  },
  plugins: [],
}
