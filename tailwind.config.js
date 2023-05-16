/** @type {import('tailwindcss').Config} */
module.exports = {
  mode: 'jit',
  content: ["./templates/**/*.{html,htm}"],
  theme: {
    extend: {
      backgroundImage: {
        plant: 'url(https://lh3.googleusercontent.com/pc64x3kOgaCI35ATNKgN0wt49L768PZFlgtsnl68vPQfW7FK6ibVvj47u8ZsBnW9suS9okklgZZMywIjq3Rlm4k8qCbDlKahRE3BX0Yh6A)'
      },
    },
  },
  plugins: [],
}