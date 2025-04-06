import{c as I}from"./chunk-LEANEOT7.js";import{h as N}from"./chunk-4SN2VY3V.js";import{h as E,i as z,j as A,n as T}from"./chunk-7EKIJKTV.js";import{Da as w,Ia as C,Oa as d,Qa as b,Sa as k,X as f,Xa as i,Y as u,Ya as r,Za as l,a as v,ab as x,b as P,db as _,eb as m,kb as s,lb as p,mb as M,ua as a,wb as S,za as y}from"./chunk-CS32D7W3.js";var D=(c,t,e,n)=>({"fa-graduation-cap":c,"fa-diagram-project":t,"fa-trophy":e,"fa-network-wired":n});function L(c,t){c&1&&(i(0,"span",32),s(1,"Popular"),r())}function j(c,t){if(c&1){let e=x();i(0,"div",28),_("mouseenter",function(){let o=f(e).$implicit,g=m(3);return u(g.setHoveredCard(o.id))})("mouseleave",function(){f(e);let o=m(3);return u(o.setHoveredCard(null))})("click",function(){let o=f(e).$implicit,g=m(3);return u(g.navigateTo(o.routePath))}),i(1,"div",29),l(2,"i"),r(),i(3,"div",30)(4,"h4"),s(5),r(),i(6,"div",31)(7,"span",32),s(8),r(),C(9,L,2,0,"span",33),r(),i(10,"p"),s(11),r()(),i(12,"div",34)(13,"div",35)(14,"span"),s(15,"Explore"),r(),l(16,"i",36),r()()()}if(c&2){let e=t.$implicit,n=m(3);b("hovered",n.hoveredCard===e.id),a(),d("ngClass","algorithm-icon-"+e.id),a(),k("fa-solid fa-",e.icon,""),a(3),p(e.name),a(3),p(n.getComplexityLevel(e.id)),a(),d("ngIf",n.isPopular(e.id)),a(2),p(e.description)}}function F(c,t){if(c&1){let e=x();i(0,"div",23)(1,"div",24),_("click",function(o){let g=f(e).$implicit,h=m(2);return u(h.toggleSubCategory(g.id,o))}),i(2,"div",25),l(3,"i",16),r(),i(4,"div",17)(5,"h3"),s(6),r(),i(7,"span",18),s(8),r()(),i(9,"div",19),l(10,"i",20),r()(),i(11,"div",26),C(12,j,17,10,"div",27),r()()}if(c&2){let e=t.$implicit,n=t.index,o=m(2);d("ngClass","subcategory-"+n),a(2),d("ngClass","subcategory-icon-"+n),a(),d("ngClass",S(10,D,e.id==="supervised-learning",e.id==="unsupervised-learning",e.id==="reinforcement-learning",e.id==="neural-networks")),a(3),p(e.name),a(2),M("",e.algorithms.length," algorithms"),a(),b("rotate",o.isSubCategoryExpanded(e.id)),a(2),b("expanded",o.isSubCategoryExpanded(e.id)),a(),d("ngForOf",e.algorithms)}}function R(c,t){if(c&1){let e=x();i(0,"div",13)(1,"div",14),_("click",function(o){let g=f(e).$implicit,h=m();return u(h.toggleCategory(g.id,o))}),i(2,"div",15),l(3,"i",16),r(),i(4,"div",17)(5,"h2"),s(6),r(),i(7,"span",18),s(8),r()(),i(9,"div",19),l(10,"i",20),r()(),i(11,"div",21),C(12,F,13,15,"div",22),r()()}if(c&2){let e=t.$implicit,n=t.index,o=m();d("ngClass","category-"+n),a(3),d("ngClass",e.id==="machine-learning"?"fa-robot":"fa-brain"),a(3),p(e.name),a(2),M("",e.subCategories.length," sub-categories"),a(),b("rotate",o.isCategoryExpanded(e.id)),a(2),b("expanded",o.isCategoryExpanded(e.id)),a(),d("ngForOf",e.subCategories)}}var V=class c{constructor(t){this.router=t}allCategories=[];categories=[];expandedCategories={};expandedSubCategories={};hoveredCard=null;searchTerm="";ngOnInit(){this.initializeData(),this.allCategories=JSON.parse(JSON.stringify(this.categories)),this.categories.length>0&&(this.expandedCategories[this.categories[0].id]=!0,this.categories[0].subCategories.forEach(t=>{this.expandedSubCategories[t.id]=!0}))}navigateTo(t){this.router.navigate([t])}toggleCategory(t,e){e.stopPropagation(),this.expandedCategories[t]=!this.expandedCategories[t];let n=this.categories.find(o=>o.id===t);this.expandedCategories[t]&&n?n.subCategories.forEach(o=>{this.expandedSubCategories[o.id]=!0}):n&&n.subCategories.forEach(o=>{this.expandedSubCategories[o.id]=!1})}toggleSubCategory(t,e){e.stopPropagation(),this.expandedSubCategories[t]=!this.expandedSubCategories[t]}isCategoryExpanded(t){return this.expandedCategories[t]||!1}isSubCategoryExpanded(t){return this.expandedSubCategories[t]||!1}setHoveredCard(t){this.hoveredCard=t}getComplexityLevel(t){return{"linear-regression":"Beginner","logistic-regression":"Beginner","decision-trees":"Intermediate","random-forests":"Advanced",svm:"Advanced","naive-bayes":"Beginner",kmeans:"Intermediate",pca:"Intermediate",autoencoders:"Advanced",qlearning:"Advanced","policy-gradient":"Expert",cnn:"Advanced",rnn:"Advanced",lstm:"Expert",gan:"Expert",transformers:"Expert"}[t]||"Intermediate"}isPopular(t){return["linear-regression","random-forests","kmeans","cnn","transformers"].includes(t)}getTotalAlgorithmsCount(){let t=0;return this.allCategories.forEach(e=>{e.subCategories.forEach(n=>{t+=n.algorithms.length})}),t}getTotalSubCategoriesCount(){let t=0;return this.allCategories.forEach(e=>{t+=e.subCategories.length}),t}onSearch(t){let e=t.target.value.toLowerCase().trim();if(this.searchTerm=e,!e){this.categories=JSON.parse(JSON.stringify(this.allCategories));return}this.categories=this.allCategories.map(n=>{let o=P(v({},n),{subCategories:[]});return o.subCategories=n.subCategories.map(g=>{let h=P(v({},g),{algorithms:[]});return h.algorithms=g.algorithms.filter(O=>O.name.toLowerCase().includes(e)||O.description.toLowerCase().includes(e)),h}).filter(g=>g.algorithms.length>0),o}).filter(n=>n.subCategories.length>0),this.categories.forEach(n=>{this.expandedCategories[n.id]=!0,n.subCategories.forEach(o=>{this.expandedSubCategories[o.id]=!0})})}initializeData(){this.categories=[{id:"machine-learning",name:"Machine Learning Algorithms",subCategories:[{id:"supervised-learning",name:"Supervised Learning",algorithms:[{id:"linear-regression",name:"Linear Regression",description:"Predicts continuous outputs (e.g., house prices) using a linear equation.",icon:"chart-line",routePath:"/linear-regression"},{id:"logistic-regression",name:"Logistic Regression",description:"Binary classification (e.g., spam detection) using the sigmoid function.",icon:"code-branch",routePath:"/logistic-regression"},{id:"decision-trees",name:"Decision Trees",description:"Classification or regression via hierarchical splits in data.",icon:"tree",routePath:"/decision-trees"},{id:"random-forests",name:"Random Forests",description:"Enhances decision trees with ensemble methods for better prediction.",icon:"sitemap",routePath:"/random-forests"},{id:"svm",name:"Support Vector Machines",description:"Classification by maximizing margin between classes.",icon:"project-diagram",routePath:"/svm"},{id:"naive-bayes",name:"Naive Bayes",description:"Classification using probabilistic independence assumptions.",icon:"pie-chart",routePath:"/naive-bayes"}]},{id:"unsupervised-learning",name:"Unsupervised Learning",algorithms:[{id:"kmeans",name:"K-means Clustering",description:"Groups data into k clusters based on similarity.",icon:"object-group",routePath:"/kmeans"},{id:"pca",name:"Principal Component Analysis",description:"Reduces dimensionality while retaining variance in data.",icon:"compress-arrows-alt",routePath:"/pca"},{id:"autoencoders",name:"Autoencoders",description:"Neural networks that learn compressed data representations.",icon:"compress",routePath:"/autoencoders"}]},{id:"reinforcement-learning",name:"Reinforcement Learning",algorithms:[{id:"qlearning",name:"Q-learning",description:"Learns optimal policies in discrete environments through rewards.",icon:"brain",routePath:"/qlearning"},{id:"policy-gradient",name:"Policy Gradient Methods",description:"Optimizes policies in continuous spaces using gradient ascent.",icon:"chart-line",routePath:"/policy-gradient"}]}]},{id:"deep-learning",name:"Deep Learning Algorithms",subCategories:[{id:"neural-networks",name:"Neural Network Models",algorithms:[{id:"cnn",name:"Convolutional Neural Networks",description:"Specialized for image processing with filters and pooling layers.",icon:"image",routePath:"/cnn"},{id:"rnn",name:"Recurrent Neural Networks",description:"Process sequences with memory of previous inputs.",icon:"sync",routePath:"/rnn"},{id:"lstm",name:"Long Short-Term Memory Networks",description:"RNNs with special gates for long-term sequence modeling.",icon:"memory",routePath:"/lstm"},{id:"gan",name:"Generative Adversarial Networks",description:"Generate realistic data through competition of two networks.",icon:"yin-yang",routePath:"/gan"},{id:"transformers",name:"Transformers",description:"NLP architecture using self-attention mechanisms.",icon:"file-alt",routePath:"/transformers"}]}]}]}static \u0275fac=function(e){return new(e||c)(y(I))};static \u0275cmp=w({type:c,selectors:[["app-algorithm-dashboard"]],decls:30,vars:4,consts:[[1,"dashboard-container"],[1,"neural-net-bg"],[1,"dashboard-header"],[1,"version-tag"],[1,"search-container"],[1,"fa-solid","fa-search"],["type","text","placeholder","Search algorithms...",1,"search-input",3,"input"],[1,"statistics-bar"],[1,"stat-item"],[1,"stat-count"],[1,"stat-label"],[1,"categories-container"],["class","category-card",3,"ngClass",4,"ngFor","ngForOf"],[1,"category-card",3,"ngClass"],[1,"category-header",3,"click"],[1,"category-icon"],[1,"fa-solid",3,"ngClass"],[1,"header-content"],[1,"header-tag"],[1,"toggle-icon"],[1,"fa-solid","fa-chevron-down"],[1,"subcategories-container"],["class","subcategory-section",3,"ngClass",4,"ngFor","ngForOf"],[1,"subcategory-section",3,"ngClass"],[1,"subcategory-header",3,"click"],[1,"subcategory-icon",3,"ngClass"],[1,"algorithms-grid"],["class","algorithm-card",3,"hovered","mouseenter","mouseleave","click",4,"ngFor","ngForOf"],[1,"algorithm-card",3,"mouseenter","mouseleave","click"],[1,"algorithm-icon",3,"ngClass"],[1,"algorithm-content"],[1,"algorithm-pill-container"],[1,"algorithm-pill"],["class","algorithm-pill",4,"ngIf"],[1,"card-footer"],[1,"view-details"],[1,"fa-solid","fa-arrow-right"]],template:function(e,n){e&1&&(i(0,"div",0),l(1,"div",1),i(2,"div",2)(3,"h1"),s(4,"AI Algorithm Explorer "),i(5,"span",3),s(6,"v2.0"),r()(),i(7,"p"),s(8,"Interactive guide to machine learning and deep learning algorithms"),r(),i(9,"div",4),l(10,"i",5),i(11,"input",6),_("input",function(g){return n.onSearch(g)}),r()()(),i(12,"div",7)(13,"div",8)(14,"div",9),s(15),r(),i(16,"div",10),s(17,"Algorithms"),r()(),i(18,"div",8)(19,"div",9),s(20),r(),i(21,"div",10),s(22,"Categories"),r()(),i(23,"div",8)(24,"div",9),s(25),r(),i(26,"div",10),s(27,"Learning Types"),r()()(),i(28,"div",11),C(29,R,13,9,"div",12),r()()),e&2&&(a(15),p(n.getTotalAlgorithmsCount()),a(5),p(n.getTotalSubCategoriesCount()),a(5),p(n.allCategories.length),a(4),d("ngForOf",n.categories))},dependencies:[T,E,z,A,N],styles:[`[_nghost-%COMP%]{display:block;min-height:100vh;background-color:#0c1428;color:#e1e7f5;font-family:Roboto,sans-serif}.dashboard-container[_ngcontent-%COMP%]{max-width:1400px;margin:0 auto;padding:2rem;position:relative;overflow:hidden}.neural-net-bg[_ngcontent-%COMP%]{position:absolute;inset:0;background-image:radial-gradient(circle at 10% 20%,rgba(66,133,244,.05) 0%,transparent 20%),radial-gradient(circle at 80% 40%,rgba(124,77,255,.05) 0%,transparent 20%),radial-gradient(circle at 30% 70%,rgba(0,201,255,.05) 0%,transparent 25%),radial-gradient(circle at 90% 90%,rgba(36,180,126,.05) 0%,transparent 15%);z-index:-1;pointer-events:none}.neural-net-bg[_ngcontent-%COMP%]:before{content:"";position:absolute;inset:0;background-image:url("data:image/svg+xml,%3Csvg width='60' height='60' viewBox='0 0 60 60' xmlns='http://www.w3.org/2000/svg'%3E%3Cg fill='none' fill-rule='evenodd'%3E%3Cg stroke='%23304978' stroke-width='1'%3E%3Cpath d='M36 34v-4h-2v4h-4v2h4v4h2v-4h4v-2h-4zm0-30V0h-2v4h-4v2h4v4h2V6h4V4h-4zM6 34v-4H4v4H0v2h4v4h2v-4h4v-2H6zM6 4V0H4v4H0v2h4v4h2V6h4V4H6z'/%3E%3C/g%3E%3C/g%3E%3C/svg%3E");opacity:.03}.dashboard-header[_ngcontent-%COMP%]{text-align:center;margin-bottom:2.5rem;position:relative}.dashboard-header[_ngcontent-%COMP%]   h1[_ngcontent-%COMP%]{font-size:3rem;font-weight:700;margin-bottom:.5rem;color:#fff;letter-spacing:-.5px;position:relative;display:inline-block}.dashboard-header[_ngcontent-%COMP%]   h1[_ngcontent-%COMP%]:after{content:"";position:absolute;bottom:-12px;left:50%;width:100px;height:4px;background:linear-gradient(90deg,#4285f4,#7c4dff);transform:translate(-50%);border-radius:2px}.dashboard-header[_ngcontent-%COMP%]   h1[_ngcontent-%COMP%]   .version-tag[_ngcontent-%COMP%]{font-size:.9rem;background:#7c4dff;color:#fff;padding:3px 8px;border-radius:12px;position:relative;top:-20px;margin-left:8px;font-weight:500}.dashboard-header[_ngcontent-%COMP%]   p[_ngcontent-%COMP%]{font-size:1.4rem;color:#8a9ab0;max-width:700px;margin:1.5rem auto 2rem;line-height:1.5;font-weight:300}.search-container[_ngcontent-%COMP%]{max-width:500px;margin:0 auto;position:relative}.search-container[_ngcontent-%COMP%]   .fa-search[_ngcontent-%COMP%]{position:absolute;left:18px;top:14px;color:#8a9ab0;font-size:16px;pointer-events:none}.search-container[_ngcontent-%COMP%]   .search-input[_ngcontent-%COMP%]{width:100%;padding:14px 20px 14px 48px;border-radius:30px;border:none;background:#1a2332;color:#e1e7f5;font-size:16px;transition:all .3s ease;box-shadow:0 4px 20px #0003}.search-container[_ngcontent-%COMP%]   .search-input[_ngcontent-%COMP%]:focus{outline:none;box-shadow:0 6px 24px #0000004d,0 0 0 2px #4285f44d;background:#232f43}.search-container[_ngcontent-%COMP%]   .search-input[_ngcontent-%COMP%]::placeholder{color:#8a9ab0}.statistics-bar[_ngcontent-%COMP%]{display:flex;justify-content:center;gap:3rem;margin-bottom:3rem}.statistics-bar[_ngcontent-%COMP%]   .stat-item[_ngcontent-%COMP%]{text-align:center;padding:1rem;background:#1a2332;border-radius:12px;width:120px;box-shadow:0 4px 20px #00000026;position:relative;overflow:hidden}.statistics-bar[_ngcontent-%COMP%]   .stat-item[_ngcontent-%COMP%]:before{content:"";position:absolute;inset:-10px;background:linear-gradient(45deg,#4285f41a,#7c4dff1a);border-radius:16px;z-index:0;transform:scale(.95);transition:transform .3s ease}.statistics-bar[_ngcontent-%COMP%]   .stat-item[_ngcontent-%COMP%]:hover:before{transform:scale(1)}.statistics-bar[_ngcontent-%COMP%]   .stat-item[_ngcontent-%COMP%]   .stat-count[_ngcontent-%COMP%]{font-size:2.2rem;font-weight:700;color:#fff;margin-bottom:.2rem;position:relative;z-index:1}.statistics-bar[_ngcontent-%COMP%]   .stat-item[_ngcontent-%COMP%]   .stat-label[_ngcontent-%COMP%]{font-size:.9rem;color:#8a9ab0;font-weight:500;position:relative;z-index:1}.categories-container[_ngcontent-%COMP%]{display:flex;flex-direction:column;gap:1.5rem}.category-card[_ngcontent-%COMP%]{background-color:#1a2332;border-radius:16px;box-shadow:0 8px 30px #0003;overflow:hidden;transition:transform .3s ease,box-shadow .3s ease;position:relative;border:1px solid rgba(255,255,255,.05)}.category-card[_ngcontent-%COMP%]:before{content:"";position:absolute;top:0;left:0;right:0;height:1px;background:linear-gradient(90deg,transparent,rgba(255,255,255,.1),transparent)}.category-card[_ngcontent-%COMP%]:hover{transform:translateY(-3px);box-shadow:0 12px 40px #00000040}.category-card.category-0[_ngcontent-%COMP%]{border-left:3px solid #4285f4}.category-card.category-0[_ngcontent-%COMP%]   .category-icon[_ngcontent-%COMP%]{background:linear-gradient(135deg,#4285f4,#0d5bdd)}.category-card.category-1[_ngcontent-%COMP%]{border-left:3px solid #7c4dff}.category-card.category-1[_ngcontent-%COMP%]   .category-icon[_ngcontent-%COMP%]{background:linear-gradient(135deg,#7c4dff,#4401ff)}.header-content[_ngcontent-%COMP%]{display:flex;flex-direction:column;flex:1;padding-right:1rem}.header-tag[_ngcontent-%COMP%]{font-size:.85rem;color:#8a9ab0;margin-top:4px;display:inline-block}.category-header[_ngcontent-%COMP%]{display:flex;justify-content:space-between;align-items:center;padding:1.5rem 2rem;cursor:pointer;-webkit-user-select:none;user-select:none;transition:background-color .2s ease;position:relative}.category-header[_ngcontent-%COMP%]:hover{background-color:#ffffff08}.category-header[_ngcontent-%COMP%]   h2[_ngcontent-%COMP%]{margin:0;font-size:1.5rem;font-weight:600;color:#fff}.category-header[_ngcontent-%COMP%]   .toggle-icon[_ngcontent-%COMP%]{display:flex;align-items:center;justify-content:center;width:40px;height:40px;border-radius:50%;background-color:#1e3a66;transition:all .3s ease}.category-header[_ngcontent-%COMP%]   .toggle-icon[_ngcontent-%COMP%]   i[_ngcontent-%COMP%]{color:#e1e7f5;transition:transform .3s ease}.category-header[_ngcontent-%COMP%]   .toggle-icon.rotate[_ngcontent-%COMP%]   i[_ngcontent-%COMP%]{transform:rotate(-180deg)}.category-header[_ngcontent-%COMP%]:hover   .toggle-icon[_ngcontent-%COMP%]{background-color:#ffffff1a}.category-header[_ngcontent-%COMP%]   .category-icon[_ngcontent-%COMP%]{width:48px;height:48px;border-radius:12px;display:flex;align-items:center;justify-content:center;margin-right:1.25rem}.category-header[_ngcontent-%COMP%]   .category-icon[_ngcontent-%COMP%]   i[_ngcontent-%COMP%]{color:#fff;font-size:22px}.subcategories-container[_ngcontent-%COMP%]{max-height:0;overflow:hidden;transition:max-height .8s cubic-bezier(0,1,0,1)}.subcategories-container.expanded[_ngcontent-%COMP%]{max-height:5000px;transition:max-height 1s cubic-bezier(.5,0,1,0)}.subcategory-section[_ngcontent-%COMP%]{border-bottom:1px solid rgba(255,255,255,.05)}.subcategory-section[_ngcontent-%COMP%]:last-child{border-bottom:none}.subcategory-section.subcategory-0[_ngcontent-%COMP%]   .subcategory-icon[_ngcontent-%COMP%]{background:linear-gradient(135deg,#4285f4,#1266f1)}.subcategory-section.subcategory-1[_ngcontent-%COMP%]   .subcategory-icon[_ngcontent-%COMP%]{background:linear-gradient(135deg,#00c9ff,#00a1cc)}.subcategory-section.subcategory-2[_ngcontent-%COMP%]   .subcategory-icon[_ngcontent-%COMP%]{background:linear-gradient(135deg,#ff6b6b,#ff3838)}.subcategory-header[_ngcontent-%COMP%]{display:flex;justify-content:space-between;align-items:center;padding:1.2rem 2rem;cursor:pointer;-webkit-user-select:none;user-select:none;transition:background-color .2s ease;background-color:#162a4a}.subcategory-header[_ngcontent-%COMP%]:hover{background-color:#1a3156}.subcategory-header[_ngcontent-%COMP%]   h3[_ngcontent-%COMP%]{margin:0;font-size:1.2rem;font-weight:500;color:#fff}.subcategory-header[_ngcontent-%COMP%]   .toggle-icon[_ngcontent-%COMP%]{display:flex;align-items:center;justify-content:center;width:32px;height:32px;border-radius:50%;background-color:#ffffff1a;transition:all .3s ease}.subcategory-header[_ngcontent-%COMP%]   .toggle-icon[_ngcontent-%COMP%]   i[_ngcontent-%COMP%]{color:#e1e7f5;transition:transform .3s ease;font-size:14px}.subcategory-header[_ngcontent-%COMP%]   .toggle-icon.rotate[_ngcontent-%COMP%]   i[_ngcontent-%COMP%]{transform:rotate(-180deg)}.subcategory-header[_ngcontent-%COMP%]:hover   .toggle-icon[_ngcontent-%COMP%]{background-color:#ffffff26}.subcategory-header[_ngcontent-%COMP%]   .subcategory-icon[_ngcontent-%COMP%]{width:36px;height:36px;border-radius:10px;display:flex;align-items:center;justify-content:center;margin-right:1rem}.subcategory-header[_ngcontent-%COMP%]   .subcategory-icon[_ngcontent-%COMP%]   i[_ngcontent-%COMP%]{color:#fff;font-size:16px}.algorithms-grid[_ngcontent-%COMP%]{display:grid;grid-template-columns:repeat(auto-fill,minmax(300px,1fr));gap:1.25rem;padding:1.5rem;background-color:#0c1428;max-height:0;overflow:hidden;transition:max-height .8s cubic-bezier(0,1,0,1)}.algorithms-grid.expanded[_ngcontent-%COMP%]{max-height:5000px;transition:max-height 1s cubic-bezier(.5,0,1,0)}.algorithm-card[_ngcontent-%COMP%]{display:flex;flex-direction:column;background-color:#1e3a66;border-radius:12px;transition:all .3s cubic-bezier(.4,0,.2,1);border:1px solid rgba(255,255,255,.05);position:relative;overflow:hidden;height:100%;transform:translateY(0);box-shadow:0 4px 15px #0000001a;cursor:pointer}.algorithm-card.hovered[_ngcontent-%COMP%]{transform:translateY(-6px) scale(1.02);box-shadow:0 12px 25px #0003;z-index:10;border-color:#4285f480}.algorithm-card.hovered[_ngcontent-%COMP%]   .view-details[_ngcontent-%COMP%]{opacity:1;transform:translate(0)}.algorithm-card.hovered[_ngcontent-%COMP%]   .algorithm-icon[_ngcontent-%COMP%]{transform:scale(1.1)}.algorithm-card[_ngcontent-%COMP%]:before{content:"";position:absolute;top:0;left:0;width:3px;height:0;background:linear-gradient(#4285f4,#7c4dff);transition:height .5s ease;z-index:1}.algorithm-card[_ngcontent-%COMP%]:hover:before{height:100%}.algorithm-icon[_ngcontent-%COMP%]{display:flex;justify-content:center;align-items:center;width:50px;height:50px;border-radius:12px;margin-bottom:1rem;flex-shrink:0;position:relative;transition:transform .3s ease,box-shadow .3s ease;background:#162a4a;box-shadow:0 4px 10px #00000026}.algorithm-icon[_ngcontent-%COMP%]:after{content:"";position:absolute;inset:0;border-radius:12px;background:linear-gradient(135deg,#ffffff1a,#fff0);z-index:0}.algorithm-icon[_ngcontent-%COMP%]   i[_ngcontent-%COMP%]{font-size:22px;color:#fff;transition:color .3s ease;position:relative;z-index:1}.algorithm-icon-linear-regression[_ngcontent-%COMP%]{background:linear-gradient(135deg,#4285f4,#0d5bdd)}.algorithm-icon-logistic-regression[_ngcontent-%COMP%]{background:linear-gradient(135deg,#1266f1,#0a47ac)}.algorithm-icon-decision-trees[_ngcontent-%COMP%]{background:linear-gradient(135deg,#00c9ff,#008db3)}.algorithm-icon-random-forests[_ngcontent-%COMP%]{background:linear-gradient(135deg,#33d4ff,#00b5e6)}.algorithm-icon-svm[_ngcontent-%COMP%]{background:linear-gradient(135deg,#7c4dff,#4401ff)}.algorithm-icon-naive-bayes[_ngcontent-%COMP%]{background:linear-gradient(135deg,#a280ff,#6934ff)}.algorithm-icon-kmeans[_ngcontent-%COMP%]{background:linear-gradient(135deg,#ff6b6b,#ff1f1f)}.algorithm-icon-pca[_ngcontent-%COMP%]{background:linear-gradient(135deg,#ff9e9e,#ff5252)}.algorithm-icon-autoencoders[_ngcontent-%COMP%]{background:linear-gradient(135deg,#24b47e,#177451)}.algorithm-icon-qlearning[_ngcontent-%COMP%]{background:linear-gradient(135deg,#35d79a,#209f6f)}.algorithm-icon-policy-gradient[_ngcontent-%COMP%]{background:linear-gradient(135deg,#339db9,#226a7d)}.algorithm-icon-cnn[_ngcontent-%COMP%]{background:linear-gradient(135deg,#5374f7,#0b39f3)}.algorithm-icon-rnn[_ngcontent-%COMP%]{background:linear-gradient(135deg,#5f69fa,#1524f7)}.algorithm-icon-lstm[_ngcontent-%COMP%]{background:linear-gradient(135deg,#6b5efc,#2613fa)}.algorithm-icon-gan[_ngcontent-%COMP%]{background:linear-gradient(135deg,#a178b0,#7b518a)}.algorithm-icon-transformers[_ngcontent-%COMP%]{background:linear-gradient(135deg,#5081bf,#345c8e)}.algorithm-content[_ngcontent-%COMP%]{flex:1;padding:1.25rem 1.25rem .5rem}.algorithm-content[_ngcontent-%COMP%]   h4[_ngcontent-%COMP%]{margin:0 0 .75rem;font-size:1.2rem;font-weight:600;color:#fff}.algorithm-content[_ngcontent-%COMP%]   p[_ngcontent-%COMP%]{margin:.75rem 0 0;font-size:.95rem;color:#8a9ab0;line-height:1.6}.algorithm-pill-container[_ngcontent-%COMP%]{display:flex;gap:.5rem;flex-wrap:wrap;margin-bottom:.5rem}.algorithm-pill[_ngcontent-%COMP%]{display:inline-block;font-size:.7rem;font-weight:600;text-transform:uppercase;letter-spacing:.5px;padding:4px 8px;border-radius:4px;background-color:#4285f426;color:#a2c3fa}.algorithm-pill[_ngcontent-%COMP%]:nth-child(2){background-color:#7c4dff26;color:#c7b3ff}.card-footer[_ngcontent-%COMP%]{padding:.75rem 1.25rem;border-top:1px solid rgba(255,255,255,.05);display:flex;justify-content:flex-end;cursor:pointer}.view-details[_ngcontent-%COMP%]{display:flex;align-items:center;color:#4285f4;font-weight:500;cursor:pointer}.view-details[_ngcontent-%COMP%]   span[_ngcontent-%COMP%]{font-size:.9rem;margin-right:8px}.view-details[_ngcontent-%COMP%]   i[_ngcontent-%COMP%]{font-size:14px;transition:transform .2s ease}.view-details[_ngcontent-%COMP%]:hover   i[_ngcontent-%COMP%]{transform:translate(3px)}@media (max-width: 992px){.dashboard-container[_ngcontent-%COMP%]{padding:1.5rem}.algorithms-grid[_ngcontent-%COMP%]{grid-template-columns:repeat(auto-fill,minmax(250px,1fr))}.statistics-bar[_ngcontent-%COMP%]{gap:1.5rem}}@media (max-width: 768px){.dashboard-header[_ngcontent-%COMP%]   h1[_ngcontent-%COMP%]{font-size:2.3rem}.dashboard-header[_ngcontent-%COMP%]   p[_ngcontent-%COMP%]{font-size:1.1rem}.statistics-bar[_ngcontent-%COMP%]{flex-wrap:wrap;justify-content:space-around}.statistics-bar[_ngcontent-%COMP%]   .stat-item[_ngcontent-%COMP%]{width:100px}.algorithms-grid[_ngcontent-%COMP%]{grid-template-columns:1fr}.category-header[_ngcontent-%COMP%], .subcategory-header[_ngcontent-%COMP%]{padding:1.2rem}}`]})};export{V as AlgorithmDashboardComponent};
