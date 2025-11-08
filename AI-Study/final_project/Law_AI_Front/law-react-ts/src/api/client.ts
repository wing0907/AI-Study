// src/api/client.ts
import axios from 'axios'

export const api = axios.create({
  baseURL: '/api',            // ✅ 프록시 사용
  timeout: 60000,
})
