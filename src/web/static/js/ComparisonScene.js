class ComparisonScene extends Phaser.Scene {
    constructor() {
        super({ key: 'ComparisonScene' });
    }

    init(data) {
        this.results  = (data && data.results)  || [];
        this.agentIds = (data && data.agentIds) || [];
        this.speed    = (data && data.speed)    || 1.0;
        this.seed     = (data && data.seed)     || 0;
    }

    preload() {
        if (!this.cache.audio.exists('click')) {
            this.load.audio('click', '/static/audio/click.mp3');
        }
    }

    create() {
        this.cameras.main.setBackgroundColor('#070714');
        const W  = this.scale.width;
        const H  = this.scale.height;
        const cx = W / 2;

        this.add.text(cx, H * 0.07, 'Results', {
            fontFamily: 'Orbitron, Arial',
            fontSize: Math.floor(Math.min(H * 0.08, 68)) + 'px',
            fontStyle: 'bold',
            color: '#00d4ff',
            shadow: { offsetX: 0, offsetY: 0, color: '#00aaff', blur: 40, fill: true },
        }).setOrigin(0.5);

        const sorted = [...this.results].sort((a, b) => b.score - a.score);
        this.buildTable(cx, W, H, sorted);

        this.makeButton(cx, H * 0.80, 'Watch Again', H, () => {
            const seed = Math.floor(Math.random() * 999999);
            this.scene.start('MultiAgentScene', {
                agents: this.agentIds,
                speed:  this.speed,
                seed,
            });
        });

        this.makeButton(cx, H * 0.91, 'Main Menu', H, () => {
            this.scene.start('MenuScene');
        });

        this.scale.on('resize', this._onResize, this);
        this.events.once('shutdown', () => this.scale.off('resize', this._onResize, this));
    }

    buildTable(cx, W, H, sorted) {
        const N      = sorted.length;
        const cols   = ['Agent', 'Score', 'Pieces', 'Lines', 'Best Combo', 'Avg/Piece'];
        const colW   = [0.22, 0.14, 0.12, 0.10, 0.16, 0.14];
        const headerFS = Math.floor(H * 0.018);
        const rowFS    = Math.floor(H * 0.022);
        const rowH     = Math.floor(H * 0.054);
        const headerH  = Math.floor(H * 0.040);
        const padX     = Math.floor(W * 0.030);

        const tableW = Math.floor(W * 0.88);
        const tableX = cx - tableW / 2;
        const tableY = Math.floor(H * 0.18);
        const tableH = headerH + N * rowH + Math.floor(rowH * 0.3);

        const gfx = this.add.graphics();
        gfx.fillStyle(0x06091a, 0.88);
        gfx.fillRoundedRect(tableX, tableY, tableW, tableH, 10);
        gfx.lineStyle(1, 0x0d2244);
        gfx.strokeRoundedRect(tableX, tableY, tableW, tableH, 10);

        const colX = colW.map((_, i) => {
            let x = tableX + padX;
            for (let j = 0; j < i; j++) x += Math.floor(colW[j] * tableW);
            return x;
        });

        let y = tableY + Math.floor(headerH * 0.3);
        cols.forEach((col, i) => {
            const align = i === 0 ? 0 : 1;
            const x = i === 0 ? colX[i] : colX[i] + Math.floor(colW[i] * tableW) - padX;
            this.add.text(x, y, col, {
                fontFamily: 'Orbitron, Arial',
                fontSize: headerFS + 'px',
                fontStyle: 'bold',
                color: '#2a5a7a',
            }).setOrigin(align, 0);
        });

        gfx.lineStyle(1, 0x0d2244);
        gfx.beginPath();
        gfx.moveTo(tableX + padX, tableY + headerH);
        gfx.lineTo(tableX + tableW - padX, tableY + headerH);
        gfx.strokePath();

        y = tableY + headerH + Math.floor(rowH * 0.15);

        sorted.forEach((entry, rank) => {
            const rowY = y + rank * rowH;
            const isWinner = rank === 0 && N > 1;
            const avg = entry.piecesPlaced > 0 ? (entry.score / entry.piecesPlaced).toFixed(1) : '0.0';

            if (isWinner) {
                gfx.fillStyle(0x0a2060, 0.60);
                gfx.fillRoundedRect(tableX + 4, rowY - Math.floor(rowH * 0.12), tableW - 8, rowH, 6);
            }

            const values = [
                entry.label,
                String(entry.score),
                String(entry.piecesPlaced),
                String(entry.linesCleared),
                '×' + entry.bestCombo,
                avg,
            ];

            values.forEach((val, i) => {
                const align = i === 0 ? 0 : 1;
                const x = i === 0 ? colX[i] : colX[i] + Math.floor(colW[i] * tableW) - padX;
                let color = '#c8e8ff';
                if (isWinner && i === 0) color = '#ffd740';
                if (isWinner && i === 1) color = '#ffd740';
                if (!isWinner && i === 0) color = '#7aaabe';

                this.add.text(x, rowY + rowH * 0.2, val, {
                    fontFamily: 'Orbitron, Arial',
                    fontSize: (i === 0 ? rowFS : Math.floor(rowFS * 0.92)) + 'px',
                    fontStyle: i === 0 ? 'bold' : 'normal',
                    color,
                }).setOrigin(align, 0);
            });

            if (isWinner) {
                this.add.text(colX[0] - Math.floor(padX * 0.5), rowY + rowH * 0.2, '▶', {
                    fontFamily: 'Orbitron, Arial',
                    fontSize: Math.floor(rowFS * 0.7) + 'px',
                    color: '#ffd740',
                }).setOrigin(1, 0);
            }
        });

        if (N === 1) {
            const entry = sorted[0];
            const subY  = tableY + tableH + Math.floor(H * 0.02);
            const subFS = Math.floor(H * 0.019);
            const hs    = getHighScore('ai', entry.agent);
            if (entry.score > hs) {
                saveHighScore('ai', entry.agent, entry.score);
                this.add.text(cx, subY, '✦ New Record for ' + entry.label + '! ✦', {
                    fontFamily: 'Orbitron, Arial',
                    fontSize: subFS + 'px',
                    fontStyle: 'bold',
                    color: '#ffd740',
                }).setOrigin(0.5, 0);
            }
        }
    }

    makeButton(x, y, label, H, callback) {
        const W  = Math.floor(H * 0.40);
        const Ht = Math.floor(H * 0.070);
        const R  = Math.floor(Ht * 0.22);
        const fs = Math.floor(H * 0.026);

        const gfx = this.add.graphics();
        this.drawBtn(gfx, x, y, W, Ht, R, false);

        const txt = this.add.text(x, y, label, {
            fontFamily: 'Orbitron, Arial',
            fontSize: fs + 'px',
            fontStyle: 'bold',
            color: '#c8e8ff',
        }).setOrigin(0.5);

        const zone = this.add.zone(x - W / 2, y - Ht / 2, W, Ht)
            .setOrigin(0, 0).setInteractive({ useHandCursor: true });
        zone.on('pointerover', () => { this.drawBtn(gfx, x, y, W, Ht, R, true); txt.setStyle({ color: '#00d4ff' }); });
        zone.on('pointerout',  () => { this.drawBtn(gfx, x, y, W, Ht, R, false); txt.setStyle({ color: '#c8e8ff' }); });
        zone.on('pointerdown', () => {
            try { this.sound.play('click', { volume: getSFXVolume() }); } catch (e) {}
            this.time.delayedCall(140, callback);
        });
    }

    drawBtn(gfx, x, y, W, H, R, hover) {
        gfx.clear();
        gfx.fillStyle(hover ? 0x0a2060 : 0x060d28);
        gfx.fillRoundedRect(x - W / 2, y - H / 2, W, H, R);
        gfx.lineStyle(2, hover ? 0x00d4ff : 0x1a4888);
        gfx.strokeRoundedRect(x - W / 2, y - H / 2, W, H, R);
    }

    _onResize() {
        clearTimeout(this._resizeTimer);
        this._resizeTimer = setTimeout(() => this.scene.restart(), 200);
    }
}
