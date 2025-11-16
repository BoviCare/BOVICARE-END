/**
 * Função helper para obter o caminho correto de arquivos públicos
 * Garante compatibilidade entre diferentes navegadores
 */
export const getPublicPath = (path) => {
  // Remove barra inicial se existir para evitar duplicação
  const cleanPath = path.startsWith('/') ? path.slice(1) : path;
  
  // process.env.PUBLIC_URL pode ser vazio em desenvolvimento
  const publicUrl = process.env.PUBLIC_URL || '';
  
  // Se PUBLIC_URL está vazio, usar caminho relativo à raiz
  if (!publicUrl) {
    return `/${cleanPath}`;
  }
  
  // Garantir que não há barras duplicadas
  const baseUrl = publicUrl.endsWith('/') ? publicUrl.slice(0, -1) : publicUrl;
  return `${baseUrl}/${cleanPath}`;
};

