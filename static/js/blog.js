// Blog Page JavaScript

document.addEventListener('DOMContentLoaded', function() {
    // Add smooth scroll behavior to blog sections
    const blogPosts = document.querySelectorAll('.blog-post');
    
    blogPosts.forEach((post, index) => {
        post.style.opacity = '0';
        post.style.transform = 'translateY(30px)';
        post.style.transition = `all 0.6s ease ${index * 0.1}s`;
        
        setTimeout(() => {
            post.style.opacity = '1';
            post.style.transform = 'translateY(0)';
        }, 100);
    });
    
    // Add copy code functionality for code blocks
    document.querySelectorAll('code').forEach(code => {
        code.style.cursor = 'pointer';
        code.title = 'Click to copy';
        
        code.addEventListener('click', function() {
            const text = this.textContent;
            navigator.clipboard.writeText(text).then(() => {
                const originalText = this.textContent;
                this.textContent = 'Copied!';
                setTimeout(() => {
                    this.textContent = originalText;
                }, 1000);
            });
        });
    });
});