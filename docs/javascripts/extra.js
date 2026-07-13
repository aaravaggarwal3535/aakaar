// docs/javascripts/extra.js
document$.subscribe(function() {
  if (!document.getElementById("floating-support-container")) {
    
    // 1. Create the floating container
    const container = document.createElement("div");
    container.id = "floating-support-container";
    
    // Style the container for a VERTICAL stack
    Object.assign(container.style, {
      position: "fixed",
      bottom: "100px",
      right: "20px",
      display: "flex",
      flexDirection: "column", 
      gap: "12px",
      zIndex: "10000",
      alignItems: "flex-end"   
    });

    // 2. Setup the Ko-fi Button 
    const kofiLink = document.createElement("a");
    kofiLink.href = "https://ko-fi.com/aarav_aggarwal"; // Update when ready
    kofiLink.target = "_blank";
    
    const kofiImg = document.createElement("img");
    kofiImg.src = "https://storage.ko-fi.com/cdn/kofi2.png?v=3"; 
    kofiImg.alt = "Support Aakaar on Ko-fi";
    kofiImg.style.height = "36px"; 
    
    kofiLink.appendChild(kofiImg);

    // 3. Setup the Smart UPI Button 
    const upiLink = document.createElement("a");
    upiLink.href = "javascript:void(0)"; // This line prevents the desktop browser error!
    upiLink.style.cursor = "pointer";
    
    const upiImg = document.createElement("img");
    upiImg.src = "https://img.shields.io/badge/UPI_Support-4CAF50?style=for-the-badge&logo=google-pay&logoColor=white";
    upiImg.alt = "Support via UPI";
    upiImg.style.height = "36px"; 
    upiImg.style.borderRadius = "4px"; 
    
    upiLink.appendChild(upiImg);

    // --- The Smart Click Logic ---
    upiLink.onclick = function(e) {
      e.preventDefault();
      
      // Check if the user is browsing on a mobile phone
      const isMobile = /Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(navigator.userAgent);
      
      if (isMobile) {
        // They are on a phone: Launch the UPI app natively!
        window.location.href = "upi://pay?pa=aaravaggarwal3535@okicici&pn=Aarav%20Aggarwal&cu=INR";
      } else {
        // They are on a computer: Copy the ID and show a premium toast notification
        navigator.clipboard.writeText("aaravaggarwal3535@okicici").then(() => {
          showToast("UPI ID Copied: aaravaggarwal3535@okicici");
        });
      }
    };

    // 4. Attach buttons to the container, then inject into the page
    container.appendChild(kofiLink);
    container.appendChild(upiLink);
    document.body.appendChild(container);
  }
});

// --- Helper Function: Premium Toast Notification ---
function showToast(message) {
  // Remove existing toast if they click the button multiple times fast
  const existing = document.getElementById("upi-toast");
  if (existing) existing.remove();

  const toast = document.createElement("div");
  toast.id = "upi-toast";
  toast.innerText = message;
  
  // Style it to look like a modern web app notification
  Object.assign(toast.style, {
    position: "fixed",
    bottom: "100px", // Floats just above your buttons
    right: "20px",
    backgroundColor: "#2c3e50",
    color: "#ffffff",
    padding: "10px 16px",
    borderRadius: "6px",
    fontFamily: "system-ui, -apple-system, sans-serif",
    fontSize: "14px",
    fontWeight: "500",
    zIndex: "10001",
    boxShadow: "0 4px 12px rgba(0,0,0,0.15)",
    opacity: "0",
    transition: "opacity 0.3s ease-in-out"
  });

  document.body.appendChild(toast);
  
  // Fade in
  setTimeout(() => toast.style.opacity = "1", 10);
  
  // Wait 3 seconds, fade out, and remove from the page
  setTimeout(() => {
    toast.style.opacity = "0";
    setTimeout(() => toast.remove(), 300);
  }, 3000);
}