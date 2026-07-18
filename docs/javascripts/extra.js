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

    // 4. Attach buttons to the container, then inject into the page
    container.appendChild(kofiLink);
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