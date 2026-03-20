# Investment Dashboard – React Frontend
# Build: docker build -t investment-dashboard ./dashboard
# Serves the Vite dev server on port 3000

FROM node:20-slim

WORKDIR /app

# Install Node dependencies first (cached layer)
COPY package.json package-lock.json* ./
RUN npm ci --silent

# Copy application source
COPY . .

EXPOSE 3000

CMD ["npm", "run", "dev", "--", "--host", "0.0.0.0"]
