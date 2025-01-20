function showSidebar(){
    event.preventDefault();
    const sidebar = document.querySelector('.sidebar');
    sidebar.style.display = 'flex';
}

function closeSidebar(){
    const sidebar = document.querySelector('.sidebar');
    sidebar.style.display = 'none';
} 

console.log("JavaScript is connected successfully!");