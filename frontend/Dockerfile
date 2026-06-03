# Stage 1: Build the Vue 3 application
FROM node:20-alpine as build-stage
WORKDIR /app
COPY package*.json ./
RUN npm install
COPY . .
RUN npm run build

# Stage 2: Serve the compiled frontend with Nginx
FROM nginx:alpine
# Copy the compiled HTML/JS/CSS files from the build stage to Nginx
COPY --from=build-stage /app/dist /usr/share/nginx/html
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
