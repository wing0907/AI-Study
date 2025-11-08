import { useEffect, useState } from 'react';

export const useLogo = () => {
  const [src, setSrc] = useState('/logo_alt.png');
  useEffect(() => {
    const img = new Image();
    img.src = '/logo_alt.png';
    img.onload = () => setSrc('/logo_alt.png');
    img.onerror = () => setSrc('/logo.png');
  }, []);
  return src;
};
